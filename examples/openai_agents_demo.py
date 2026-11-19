"""The SAME clinic idea, written with the OpenAI Agents SDK, to compare with LangGraph.

Run (needs an OpenAI key):   pip install openai-agents
                             set OPENAI_API_KEY=sk-...        (Windows cmd)
                             python examples/openai_agents_demo.py

Mapping to what you already know:
    LangGraph supervisor node + routing edges  ->  a triage Agent with `handoffs=[...]`
    @tool (langchain_core)                     ->  @function_tool
    graph.invoke(...) + checkpointer           ->  Runner.run_sync(...) + you pass the history yourself
Note: this demo has NO confirmation pause before booking. LangGraph's interrupt() is what gave us that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Agent, Runner, function_tool

from app.config import settings
from app.db import ClinicDB
from app.embeddings import get_embeddings
from app.knowledge import KnowledgeBase

MODEL = "gpt-4o-mini"


def build_agents(db: ClinicDB, kb: KnowledgeBase) -> Agent:
    @function_tool
    def search_clinic_info(question: str) -> str:
        """Search the clinic documents (hours, fees, services, policies)."""
        hits = [(d, s) for d, s in kb.search(question, k=3) if s >= 0.15]
        return "\n\n".join(f"[source: {d.metadata['source']}] {d.page_content}" for d, _ in hits) or "NO_RELEVANT_INFO"

    @function_tool
    def list_available_slots(specialty: str) -> str:
        """List free appointment slots for a specialty."""
        slots = db.list_available_slots(specialty)
        return "\n".join(f"slot_id={s['slot_id']}: {s['starts_at']} with {s['doctor']}" for s in slots) or "NO_SLOTS"

    @function_tool
    def register_and_book(full_name: str, phone: str, date_of_birth: str, slot_id: int) -> str:
        """Register the patient (if new) and book the slot. date_of_birth is YYYY-MM-DD."""
        patient, _ = db.create_patient(full_name, phone, date_of_birth)
        code = db.book_slot(patient["id"], slot_id)
        return f"BOOKED: confirmation_code={code}"

    info = Agent(
        name="Info specialist", model=MODEL, tools=[search_clinic_info],
        handoff_description="Answers questions about the clinic (hours, fees, services, policies).",
        instructions="Answer only from search_clinic_info and mention the source file. Never give medical advice.",
    )
    booking = Agent(
        name="Booking specialist", model=MODEL, tools=[list_available_slots, register_and_book],
        handoff_description="Registers patients and books appointments.",
        instructions="Show slots with list_available_slots, collect name, phone, date of birth, then register_and_book.",
    )
    return Agent(
        name="Front desk", model=MODEL, handoffs=[info, booking],
        instructions="Greet the patient and hand off to the right specialist. Do not answer yourself.",
    )


if __name__ == "__main__":
    db = ClinicDB("examples_demo.db")
    db.seed()
    kb = KnowledgeBase(get_embeddings(settings), "examples_vectors", collection="demo_docs")
    if kb.count() == 0:
        kb.ingest(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    front_desk = build_agents(db, kb)
    history = []
    print("Type a message (Ctrl+C to quit).")
    while True:
        history.append({"role": "user", "content": input("You: ")})
        result = Runner.run_sync(front_desk, history)
        print(f"[{result.last_agent.name}] {result.final_output}")
        history = result.to_input_list()
