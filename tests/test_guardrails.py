from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.guardrails import PatientInput, is_emergency, normalize_phone


@pytest.mark.parametrize("text", [
    "I have severe chest pain", "my father can't breathe", "she fainted at home", "I want to kill myself",
])
def test_emergencies_are_detected(text):
    assert is_emergency(text)


@pytest.mark.parametrize("text", ["What are your opening hours?", "Book me with a dermatologist"])
def test_normal_messages_pass(text):
    assert not is_emergency(text)


@pytest.mark.parametrize("raw", ["03001234567", "0300-1234567", "+923001234567", "0300 1234567"])
def test_phone_numbers_are_normalised(raw):
    assert normalize_phone(raw) == "03001234567"


@pytest.mark.parametrize("raw", ["12345", "04001234567", "0300123456a"])
def test_bad_phone_numbers_are_rejected(raw):
    with pytest.raises(ValueError):
        normalize_phone(raw)


def test_valid_patient_is_cleaned():
    p = PatientInput(full_name="  ali   raza ", phone="0300-1234567", date_of_birth="1998-05-10")
    assert p.full_name == "Ali Raza" and p.phone == "03001234567"


def test_future_date_of_birth_is_rejected():
    future = (date.today() + timedelta(days=10)).isoformat()
    with pytest.raises(ValidationError):
        PatientInput(full_name="Ali", phone="03001234567", date_of_birth=future)
