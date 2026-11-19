"""REST API around the agent.

POST /chat                      send a message (creates a conversation if no thread_id is given)
POST /chat/{thread_id}/confirm  answer a pending booking confirmation (approve or decline)
POST /ingest                    re-run the document ETL pipeline
GET  /health                    liveness check
GET  /                          a tiny chat page for demos
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import ClinicAgent, build_graph
from app.multi_agent import build_multi_graph
from app.config import BASE_DIR, settings
from app.db import ClinicDB
from app.embeddings import get_embeddings
from app.knowledge import KnowledgeBase
from app.llm import get_llm
from app.tools import build_tools


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


class ConfirmRequest(BaseModel):
    approved: bool


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    pending_confirmation: dict | None = None
    handled_by: str | None = None  # "info" or "booking" in multi-agent mode


def build_services() -> tuple[ClinicAgent, KnowledgeBase]:
    db = ClinicDB(settings.database_path)
    db.seed()
    kb = KnowledgeBase(get_embeddings(settings), settings.vector_dir,
                       collection=f"clinic_docs_{settings.embeddings_provider}")
    if kb.count() == 0:
        kb.ingest(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    build = build_multi_graph if settings.agent_mode == "multi" else build_graph
    graph = build(get_llm(settings), build_tools(db, kb, settings.top_k))
    return ClinicAgent(graph), kb


def create_app(agent: ClinicAgent | None = None, kb: KnowledgeBase | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            app.state.agent, app.state.kb = build_services()
        else:
            app.state.agent, app.state.kb = agent, kb
        yield

    app = FastAPI(title="Clinic Front-Desk Agent", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        thread_id = req.thread_id or str(uuid.uuid4())
        try:
            return app.state.agent.chat(thread_id, req.message)
        except RuntimeError as e:  # a confirmation is still pending
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/chat/{thread_id}/confirm", response_model=ChatResponse)
    def confirm(thread_id: str, req: ConfirmRequest):
        try:
            return app.state.agent.confirm(thread_id, req.approved)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/ingest")
    def ingest():
        if app.state.kb is None:
            raise HTTPException(status_code=503, detail="Knowledge base not configured.")
        return app.state.kb.ingest(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(BASE_DIR) / "static" / "index.html")

    return app


app = create_app()
