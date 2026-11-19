def test_register_patient_creates_record(tools, db):
    out = tools["register_patient"].invoke(
        {"full_name": "Sara Khan", "phone": "0321-5550000", "date_of_birth": "1995-02-01"})
    assert out.startswith("REGISTERED")
    assert db.find_patient_by_phone("03215550000")["full_name"] == "Sara Khan"


def test_register_patient_never_creates_duplicates(tools, db):
    args = {"full_name": "Sara Khan", "phone": "03215550000", "date_of_birth": "1995-02-01"}
    tools["register_patient"].invoke(args)
    second = tools["register_patient"].invoke(args)
    assert second.startswith("ALREADY_REGISTERED")


def test_register_patient_reports_the_bad_field(tools):
    out = tools["register_patient"].invoke(
        {"full_name": "Sara Khan", "phone": "12345", "date_of_birth": "1995-02-01"})
    assert out.startswith("ERROR") and "phone" in out


def test_find_patient(tools):
    assert tools["find_patient"].invoke({"phone": "03009999999"}).startswith("NOT_FOUND")
    tools["register_patient"].invoke({"full_name": "Omar", "phone": "03009999999", "date_of_birth": "1990-01-01"})
    assert "name=Omar" in tools["find_patient"].invoke({"phone": "0300-9999999"})


def test_list_slots_rejects_unknown_specialty(tools):
    out = tools["list_available_slots"].invoke({"specialty": "Astrology"})
    assert out.startswith("ERROR") and "Dermatology" in out


def test_list_slots_returns_free_slots(tools):
    out = tools["list_available_slots"].invoke({"specialty": "dermatology"})
    assert "Dr. Sana Malik" in out and "slot_id=" in out


def test_search_returns_sources(tools):
    out = tools["search_clinic_info"].invoke({"question": "Do I need fasting before a lipid profile test?"})
    assert "source: 04_test_preparation.md" in out


def test_double_booking_is_impossible(db):
    patient, _ = db.create_patient("Ali", "03001112222", "1990-01-01")
    slot_id = db.list_available_slots("Cardiology")[0]["slot_id"]
    db.book_slot(patient["id"], slot_id)
    try:
        db.book_slot(patient["id"], slot_id)
        raise AssertionError("second booking should fail")
    except ValueError:
        pass
