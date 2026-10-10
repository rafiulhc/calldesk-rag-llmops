import os
import secrets
from contextlib import asynccontextmanager

import openai
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ask import ask
from db import close_pool, get_pool
from ingest import EMBEDDING_MODEL, client

API_KEY = os.environ["APP_API_KEY"]   # no fallback: refuse to start without it


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
    yield
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