"""Shared test fixtures. The LLM is replaced by a scripted fake, so tests are fast,
free, deterministic, and need no API key."""

from datetime import date, timedelta

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import ClinicAgent, build_graph
from app.config import BASE_DIR
from app.db import ClinicDB
from app.embeddings import LocalHashingEmbeddings
from app.knowledge import KnowledgeBase
from app.tools import build_tools

DOCS_DIR = str(BASE_DIR / "data" / "clinic_docs")


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AI messages in order, and records what it was sent."""

    responses: list = []
    received: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.responses:
            raise AssertionError("The LLM was called more times than the test expected.")
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def db(tmp_path) -> ClinicDB:
    database = ClinicDB(str(tmp_path / "test.db"))
    database.seed(start=date.today() + timedelta(days=1))
    return database


@pytest.fixture
def kb(tmp_path) -> KnowledgeBase:
    knowledge = KnowledgeBase(LocalHashingEmbeddings(), str(tmp_path / "vectors"), collection="test_docs")
    knowledge.ingest(DOCS_DIR)
    return knowledge


@pytest.fixture
def tools(db, kb):
    return {t.name: t for t in build_tools(db, kb)}


@pytest.fixture
def make_agent(db, kb):
    def _make(responses: list) -> tuple[ClinicAgent, ScriptedChatModel]:
        llm = ScriptedChatModel(responses=list(responses), received=[])
        return ClinicAgent(build_graph(llm, build_tools(db, kb))), llm

    return _make
