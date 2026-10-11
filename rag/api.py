import hashlib
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any

import openai
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import rate_limit
from agent_service import close_agent, init_agent, run_agent
from ask import ask
from db import close_pool, get_pool
from ingest import EMBEDDING_MODEL, client

API_KEY = os.environ["APP_API_KEY"]   # no fallback: refuse to start without it
KEY_ID = hashlib.sha256(API_KEY.encode()).hexdigest()[:12]   # scopes threads to the client's key


@asynccontextmanager
async def lifespan(app):
    """Warm up at startup so the first real request isn't slow; clean up at shutdown."""
    with get_pool().connection() as conn:     # open the pool now (and fail fast if the DB is unreachable)
        conn.execute("SELECT 1")
    try:
        # Warm the endpoint the app actually uses. Measured: the first embedding calls after startup take
        # ~700-1,300 ms, then settle at ~260 ms; warming a different endpoint (models.retrieve) didn't help.
        for _ in range(2):
            client.embeddings.create(model=EMBEDDING_MODEL, input=["warm-up"])   # ~1 token each, ~$0.00000002
    except openai.APIError:
        pass                                      # warm-up only; requests will still work or report errors
    init_agent()                                  # checkpointer pool + tables for /agent conversations
    yield
    close_agent()
    close_pool()


app = FastAPI(title="Calldesk RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Source(BaseModel):
    source: str
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def require_api_key(x_api_key: str = Header(default="")):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    wait = rate_limit.limiter.check(KEY_ID)          # only counted after the key is valid
    if wait:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": str(wait)})


@app.get("/health")
def health():
    with get_pool().connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask_endpoint(body: AskRequest):
    try:
        answer, results = ask(body.question)
    except openai.APIError:
        raise HTTPException(status_code=503, detail="The language model is unavailable. Please try again.")
    sources = [Source(source=c["source"], chunk_id=c["chunk_id"], score=round(s, 3)) for s, c in results]
    return AskResponse(answer=answer, sources=sources)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    thread_id: uuid.UUID | None = None     # omit to start a new conversation; must be a UUID (unguessable)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]


class AgentResponse(BaseModel):
    answer: str
    thread_id: uuid.UUID
    tool_calls: list[ToolCall]


@app.post("/agent", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def agent_endpoint(body: AgentRequest):
    thread_id = body.thread_id or uuid.uuid4()
    try:
        answer, tool_calls = run_agent(body.message, f"{KEY_ID}:{thread_id}")
    except openai.APIError:
        raise HTTPException(status_code=503, detail="The language model is unavailable. Please try again.")
    return AgentResponse(answer=answer, thread_id=thread_id, tool_calls=tool_calls)