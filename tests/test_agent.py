from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import tool_call


def test_agent_uses_a_tool_then_answers(make_agent):
    agent, llm = make_agent([
        tool_call("search_clinic_info", {"question": "opening hours"}),
        AIMessage(content="We're open Monday to Friday, 9 AM to 6 PM. (source: 01_about_and_hours.md)"),
    ])
    result = agent.chat("t1", "When are you open?")
    assert "9 AM to 6 PM" in result["reply"]
    assert result["pending_confirmation"] is None
    # The second LLM call must have received the tool's output (the ReAct loop)
    tool_msgs = [m for m in llm.received[1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "01_about_and_hours.md" in tool_msgs[0].content


def test_emergency_never_reaches_the_llm(make_agent):
    agent, llm = make_agent([])  # the LLM would raise if called
    result = agent.chat("t2", "My mother has chest pain and can't breathe")
    assert "emergency" in result["reply"].lower()
    assert llm.received == []


def test_booking_waits_for_confirmation_then_books(make_agent, db):
    patient, _ = db.create_patient("Ali Raza", "03001234567", "1998-05-10")
    slot_id = db.list_available_slots("General Physician")[0]["slot_id"]
    agent, _ = make_agent([
        tool_call("book_appointment", {"patient_id": patient["id"], "slot_id": slot_id}),
        AIMessage(content="Done! Your appointment is booked."),
    ])

    first = agent.chat("t3", "Book that slot please")
    assert first["pending_confirmation"]["type"] == "confirm_booking"
    assert db.get_slot(slot_id)["is_booked"] == 0  # nothing saved yet

    final = agent.confirm("t3", approved=True)
    assert final["pending_confirmation"] is None
    assert db.get_slot(slot_id)["is_booked"] == 1
    assert len(db.list_appointments(patient["id"])) == 1


def test_declined_booking_saves_nothing(make_agent, db):
    patient, _ = db.create_patient("Ali Raza", "03001234567", "1998-05-10")
    slot_id = db.list_available_slots("Pediatrics")[0]["slot_id"]
    agent, _ = make_agent([
        tool_call("book_appointment", {"patient_id": patient["id"], "slot_id": slot_id}),
        AIMessage(content="No problem, I haven't booked it."),
    ])
    agent.chat("t4", "Book it")
    agent.confirm("t4", approved=False)
    assert db.get_slot(slot_id)["is_booked"] == 0


def test_conversation_memory_per_thread(make_agent):
    agent, llm = make_agent([AIMessage(content="Hi Ali!"), AIMessage(content="Your name is Ali.")])
    agent.chat("t5", "My name is Ali")
    agent.chat("t5", "What is my name?")
    second_call = [m.content for m in llm.received[1] if isinstance(m, HumanMessage)]
    assert "My name is Ali" in second_call  # the earlier message was remembered


def test_threads_do_not_share_memory(make_agent):
    agent, llm = make_agent([AIMessage(content="Hi Ali!"), AIMessage(content="I don't know your name.")])
    agent.chat("thread-a", "My name is Ali")
    agent.chat("thread-b", "What is my name?")
    other_thread = [m.content for m in llm.received[1] if isinstance(m, HumanMessage)]
    assert "My name is Ali" not in other_thread
