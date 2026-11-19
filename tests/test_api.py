from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import create_app
from conftest import tool_call


def client_for(agent, kb):
    return TestClient(create_app(agent=agent, kb=kb))


def test_health(make_agent, kb):
    agent, _ = make_agent([])
    with client_for(agent, kb) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_chat_creates_a_thread(make_agent, kb):
    agent, _ = make_agent([AIMessage(content="Hello! How can I help?")])
    with client_for(agent, kb) as client:
        res = client.post("/chat", json={"message": "Hi"})
        assert res.status_code == 200
        body = res.json()
        assert body["thread_id"] and body["reply"] == "Hello! How can I help?"


def test_empty_message_is_rejected(make_agent, kb):
    agent, _ = make_agent([])
    with client_for(agent, kb) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422


def test_confirm_flow_and_status_codes(make_agent, kb, db):
    patient, _ = db.create_patient("Ali Raza", "03001234567", "1998-05-10")
    slot_id = db.list_available_slots("Cardiology")[0]["slot_id"]
    agent, _ = make_agent([
        tool_call("book_appointment", {"patient_id": patient["id"], "slot_id": slot_id}),
        AIMessage(content="Booked!"),
    ])
    with client_for(agent, kb) as client:
        assert client.post("/chat/none/confirm", json={"approved": True}).status_code == 404

        body = client.post("/chat", json={"message": "book it", "thread_id": "x1"}).json()
        assert body["pending_confirmation"]["slot_id"] == slot_id

        # New messages are refused while a confirmation is pending
        assert client.post("/chat", json={"message": "hello", "thread_id": "x1"}).status_code == 409

        done = client.post("/chat/x1/confirm", json={"approved": True}).json()
        assert done["reply"] == "Booked!" and done["pending_confirmation"] is None
