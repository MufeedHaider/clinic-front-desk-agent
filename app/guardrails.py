"""Guardrails: checks around the AI so one mistake by the model can't cause harm.

1. Input guardrail: emergencies are answered with a fixed safety message, never by the LLM.
2. Validation: every value the LLM passes to a tool is checked on the server.
"""

import re
from datetime import date

from pydantic import BaseModel, Field, ValidationError, field_validator

EMERGENCY_PATTERNS = [
    r"chest pain", r"can'?t breathe", r"cannot breathe", r"difficulty breathing", r"short(ness)? of breath",
    r"unconscious", r"fainted", r"severe bleeding", r"bleeding heavily", r"seizure", r"stroke",
    r"heart attack", r"overdose", r"suicid", r"kill myself", r"poison",
    # Roman Urdu (the clinic's patients also write this way)
    r"seene\s+m(ein|e)\s+dard", r"s(a|aa)ns\s+n(ahi|ahin)", r"behosh", r"dil\s+ka\s+daura",
    r"khoon\s+(beh|bah)", r"zeher", r"khud\s?kushi",
]
EMERGENCY_REPLY = (
    "This sounds like it could be an emergency. Please call your local emergency number or go to the "
    "nearest hospital emergency department right now. This assistant can't help with emergencies, "
    "and the clinic does not provide emergency care."
)


def is_emergency(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in EMERGENCY_PATTERNS)


class PatientInput(BaseModel):
    full_name: str = Field(min_length=2, max_length=80)
    phone: str
    date_of_birth: date

    @field_validator("full_name")
    @classmethod
    def name_letters_only(cls, v: str) -> str:
        v = " ".join(v.split())
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", v):
            raise ValueError("name should contain letters only")
        return v.title()

    @field_validator("phone")
    @classmethod
    def pakistani_mobile(cls, v: str) -> str:
        return normalize_phone(v)

    @field_validator("date_of_birth")
    @classmethod
    def realistic_dob(cls, v: date) -> date:
        today = date.today()
        if v > today:
            raise ValueError("date of birth cannot be in the future")
        if today.year - v.year > 120:
            raise ValueError("date of birth is not realistic")
        return v


def normalize_phone(v: str) -> str:
    """Accept 03XX-XXXXXXX, 03XXXXXXXXX or +923XXXXXXXXX and return 03XXXXXXXXX."""
    digits = re.sub(r"[\s-]", "", v)
    if digits.startswith("+92"):
        digits = "0" + digits[3:]
    elif digits.startswith("92") and len(digits) == 12:
        digits = "0" + digits[2:]
    if not re.fullmatch(r"03\d{9}", digits):
        raise ValueError("phone must be a Pakistani mobile number like 03001234567")
    return digits


def friendly_errors(err: ValidationError) -> str:
    """Turn a Pydantic error into a short message the agent can relay field by field."""
    parts = []
    for e in err.errors():
        field = ".".join(str(x) for x in e["loc"]) or "input"
        parts.append(f"{field}: {e['msg'].removeprefix('Value error, ')}")
    return "; ".join(parts)
