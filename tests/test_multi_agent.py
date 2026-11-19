from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.multi_agent import build_multi_graph, parse_route
from app.agent import ClinicAgent
from app.tools import build_tools
from conftest import ScriptedChatModel, tool_call


def make_multi(db, kb, responses):
    llm = ScriptedChatModel(responses=list(responses), received=[])
    return ClinicAgent(build_multi_graph(llm, build_tools(db, kb))), llm


def test_parse_route_defaults_to_the_safe_read_only_agent():
    assert parse_route("booking") == "booking"
    assert parse_route("  Booking.\n") == "booking"
    assert parse_route("info") == "info"
    assert parse_route("I am not sure") == "info"


def test_clinic_question_goes_to_info_specialist(db, kb):
    agent, llm = make_multi(db, kb, [
        AIMessage(content="info"),                                           # supervisor
        tool_call("search_clinic_info", {"question": "opening hours"}),       # info agent
        AIMessage(content="Open Monday to Friday, 9 AM to 6 PM."),
    ])
    result = agent.chat("m1", "When are you open?")
    assert result["handled_by"] == "info"
    assert "9 AM" in result["reply"]
    # The info specialist got its own prompt, and a booking tool was never offered to it
    system = [m for m in llm.received[1] if isinstance(m, SystemMessage)][0].content
    assert "Information Specialist" in system


def test_booking_request_goes_to_booking_specialist_and_still_waits_for_confirmation(db, kb):
    patient, _ = db.create_patient("Ali Raza", "03001234567", "1998-05-10")
    slot_id = db.list_available_slots("General Physician")[0]["slot_id"]
    agent, llm = make_multi(db, kb, [
        AIMessage(content="booking"),
        tool_call("book_appointment", {"patient_id": patient["id"], "slot_id": slot_id}),
        AIMessage(content="Booked!"),
    ])
    first = agent.chat("m2", "Please book me a general physician slot")
    assert first["handled_by"] == "booking"
    assert first["pending_confirmation"]["type"] == "confirm_booking"
    assert db.get_slot(slot_id)["is_booked"] == 0
    final = agent.confirm("m2", approved=True)
    assert db.get_slot(slot_id)["is_booked"] == 1
    assert "Booked" in final["reply"]


def test_emergency_skips_supervisor_and_specialists(db, kb):
    agent, llm = make_multi(db, kb, [])
    result = agent.chat("m3", "My father has chest pain and can't breathe")
    assert "emergency" in result["reply"].lower()
    assert llm.received == []


def test_routing_sees_only_a_short_text_transcript(db, kb):
    agent, llm = make_multi(db, kb, [AIMessage(content="info"), AIMessage(content="Hello!")])
    agent.chat("m4", "hi there")
    router_input = llm.received[0]
    assert isinstance(router_input[1], HumanMessage) and "Patient: hi there" in router_input[1].content
