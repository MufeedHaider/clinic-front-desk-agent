"""The agent's tools. The LLM decides WHEN to call them and with WHAT arguments;
this code decides whether the call is valid, and does the real work.
"""

from datetime import date

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import ValidationError

from app.db import ClinicDB
from app.guardrails import PatientInput, friendly_errors, normalize_phone
from app.knowledge import KnowledgeBase

MIN_RELEVANCE = 0.15  # below this, a retrieved chunk is treated as "not relevant"


def build_tools(db: ClinicDB, kb: KnowledgeBase, top_k: int = 3) -> list:
    @tool
    def search_clinic_info(question: str) -> str:
        """Search the clinic's documents (hours, fees, services, policies, test preparation).
        Use this for any question about the clinic. Returns text passages with their source file."""
        results = [(d, s) for d, s in kb.search(question, k=top_k) if s >= MIN_RELEVANCE]
        if not results:
            return "NO_RELEVANT_INFO: nothing in the clinic documents answers this."
        return "\n\n".join(f"[source: {d.metadata['source']}] {d.page_content}" for d, _ in results)

    @tool
    def find_patient(phone: str) -> str:
        """Look up a registered patient by mobile phone number (e.g. 03001234567)."""
        try:
            phone = normalize_phone(phone)
        except ValueError as e:
            return f"ERROR: phone: {e}"
        p = db.find_patient_by_phone(phone)
        if not p:
            return f"NOT_FOUND: no patient registered with phone {phone}."
        appts = db.list_appointments(p["id"])
        upcoming = "; ".join(f"{a['starts_at']} with {a['doctor']} ({a['confirmation_code']})" for a in appts)
        return (f"FOUND: patient_id={p['id']}, name={p['full_name']}, date_of_birth={p['date_of_birth']}. "
                f"Appointments: {upcoming or 'none'}.")

    @tool
    def register_patient(full_name: str, phone: str, date_of_birth: str) -> str:
        """Register a new patient. date_of_birth must be YYYY-MM-DD.
        If the phone number is already registered, the existing patient is returned (no duplicate)."""
        try:
            data = PatientInput(full_name=full_name, phone=phone, date_of_birth=date_of_birth)
        except ValidationError as e:
            return f"ERROR: {friendly_errors(e)}"
        patient, created = db.create_patient(data.full_name, data.phone, data.date_of_birth.isoformat())
        if not created:
            return (f"ALREADY_REGISTERED: patient_id={patient['id']}, name={patient['full_name']}. "
                    "No duplicate was created.")
        return f"REGISTERED: patient_id={patient['id']}, name={patient['full_name']}."

    @tool
    def list_available_slots(specialty: str, on_date: str | None = None) -> str:
        """List free appointment slots for a specialty (e.g. 'General Physician', 'Pediatrics',
        'Dermatology', 'Cardiology'). Optionally filter by date (YYYY-MM-DD)."""
        known = db.list_specialties()
        if specialty.lower() not in {s.lower() for s in known}:
            return f"ERROR: specialty: unknown. Available specialties: {', '.join(known)}."
        if on_date:
            try:
                date.fromisoformat(on_date)
            except ValueError:
                return "ERROR: on_date: use the format YYYY-MM-DD."
        slots = db.list_available_slots(specialty, on_date)
        if not slots:
            return "NO_SLOTS: no free slots match. Try another date."
        return "\n".join(f"slot_id={s['slot_id']}: {s['starts_at']} with {s['doctor']}" for s in slots)

    @tool
    def book_appointment(patient_id: int, slot_id: int) -> str:
        """Book a slot for a registered patient. The patient is asked to confirm before anything is saved."""
        patient = db.get_patient(patient_id)
        if not patient:
            return "ERROR: patient_id: no such patient. Register the patient first."
        slot = db.get_slot(slot_id)
        if not slot:
            return "ERROR: slot_id: no such slot. List available slots first."
        if slot["is_booked"]:
            return "ERROR: slot_id: this slot is already booked. Offer another slot."

        # Human-in-the-loop: pause the whole workflow here until the patient confirms.
        decision = interrupt({
            "type": "confirm_booking",
            "question": f"Book {slot['starts_at']} with {slot['doctor']} ({slot['specialty']}) "
                        f"for {patient['full_name']}?",
            "patient_id": patient_id,
            "slot_id": slot_id,
        })
        if not (isinstance(decision, dict) and decision.get("approved")):
            return ("CANCELLED: the patient said No, so nothing was booked. Do not ask them to confirm again. "
                    "Tell them it is cancelled and ask if they want a different slot.")
        try:
            code = db.book_slot(patient_id, slot_id)
        except ValueError as e:
            return f"ERROR: slot_id: {e}"
        return (f"BOOKED: confirmation_code={code}, {slot['starts_at']} with {slot['doctor']}. "
                "Remind the patient to arrive 15 minutes early.")

    return [search_clinic_info, find_patient, register_patient, list_available_slots, book_appointment]
