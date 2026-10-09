import os
import secrets

import openai
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ask import ask
from db import get_pool

API_KEY = os.environ["APP_API_KEY"]   # no fallback: refuse to start without it

app = FastAPI(title="Calldesk RAG API")


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