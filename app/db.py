"""SQLite storage for patients, doctors, appointment slots and bookings.

Uses plain SQL so every query is visible and easy to explain.
"""

import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT    NOT NULL,
    phone         TEXT    NOT NULL UNIQUE,          -- one record per phone number
    date_of_birth TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS doctors (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    specialty TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL REFERENCES doctors(id),
    starts_at TEXT    NOT NULL,
    is_booked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS appointments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        INTEGER NOT NULL REFERENCES patients(id),
    slot_id           INTEGER NOT NULL UNIQUE REFERENCES slots(id),  -- a slot can be booked once
    confirmation_code TEXT    NOT NULL UNIQUE,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_slots_doctor_time ON slots(doctor_id, starts_at);
"""

SEED_DOCTORS = [
    ("Dr. Ayesha Khan", "General Physician"),
    ("Dr. Bilal Ahmed", "Pediatrics"),
    ("Dr. Sana Malik", "Dermatology"),
    ("Dr. Omar Siddiqui", "Cardiology"),
]
SEED_TIMES = ["10:00", "11:00", "15:00"]


class ClinicDB:
    def __init__(self, path: str):
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- setup ----------
    def seed(self, start: date | None = None, days: int = 5) -> None:
        """Add sample doctors and slots for the next `days` weekdays (only if empty)."""
        start = start or date.today() + timedelta(days=1)
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] > 0:
                return
            conn.executemany("INSERT INTO doctors (name, specialty) VALUES (?, ?)", SEED_DOCTORS)
            doctor_ids = [r["id"] for r in conn.execute("SELECT id FROM doctors")]
            day, added = start, 0
            while added < days:
                if day.weekday() < 5:  # Monday–Friday
                    for doctor_id in doctor_ids:
                        for t in SEED_TIMES:
                            conn.execute(
                                "INSERT INTO slots (doctor_id, starts_at) VALUES (?, ?)",
                                (doctor_id, f"{day.isoformat()} {t}"),
                            )
                    added += 1
                day += timedelta(days=1)

    # ---------- patients ----------
    def find_patient_by_phone(self, phone: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM patients WHERE phone = ?", (phone,)).fetchone()
        return dict(row) if row else None

    def get_patient(self, patient_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return dict(row) if row else None

    def create_patient(self, full_name: str, phone: str, date_of_birth: str) -> tuple[dict, bool]:
        """Return (patient, created). If the phone already exists, return the existing patient."""
        existing = self.find_patient_by_phone(phone)
        if existing:
            return existing, False
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO patients (full_name, phone, date_of_birth, created_at) VALUES (?, ?, ?, ?)",
                (full_name, phone, date_of_birth, datetime.now().isoformat(timespec="seconds")),
            )
            patient_id = cur.lastrowid
        return self.get_patient(patient_id), True

    # ---------- doctors & slots ----------
    def list_specialties(self) -> list[str]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT specialty FROM doctors ORDER BY specialty")]

    def list_available_slots(self, specialty: str, on_date: str | None = None, limit: int = 8) -> list[dict]:
        sql = """
            SELECT s.id AS slot_id, s.starts_at, d.name AS doctor, d.specialty
            FROM slots s
            JOIN doctors d ON d.id = s.doctor_id
            WHERE s.is_booked = 0
              AND LOWER(d.specialty) = LOWER(?)
        """
        params: list = [specialty]
        if on_date:
            sql += " AND DATE(s.starts_at) = ?"
            params.append(on_date)
        sql += " ORDER BY s.starts_at LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def get_slot(self, slot_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT s.id AS slot_id, s.starts_at, s.is_booked, d.name AS doctor, d.specialty
                   FROM slots s JOIN doctors d ON d.id = s.doctor_id WHERE s.id = ?""",
                (slot_id,),
            ).fetchone()
        return dict(row) if row else None

    def book_slot(self, patient_id: int, slot_id: int) -> str:
        """Book a slot inside one transaction. Returns a confirmation code.

        The UPDATE only succeeds if the slot is still free, so two people can't book it at once.
        """
        code = "CL-" + secrets.token_hex(3).upper()
        with self._connect() as conn:
            cur = conn.execute("UPDATE slots SET is_booked = 1 WHERE id = ? AND is_booked = 0", (slot_id,))
            if cur.rowcount == 0:
                raise ValueError("That slot is no longer available.")
            conn.execute(
                "INSERT INTO appointments (patient_id, slot_id, confirmation_code, created_at) VALUES (?, ?, ?, ?)",
                (patient_id, slot_id, code, datetime.now().isoformat(timespec="seconds")),
            )
        return code

    def list_appointments(self, patient_id: int) -> list[dict]:
        with self._connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    """SELECT a.confirmation_code, s.starts_at, d.name AS doctor, d.specialty
                       FROM appointments a
                       JOIN slots s   ON s.id = a.slot_id
                       JOIN doctors d ON d.id = s.doctor_id
                       WHERE a.patient_id = ?
                       ORDER BY s.starts_at""",
                    (patient_id,),
                )
            ]
