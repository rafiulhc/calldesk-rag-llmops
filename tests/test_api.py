from contextlib import contextmanager

import openai
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)
HEADERS = {"X-API-Key": "test-api-key"}


def fake_ask(question):
    return "899 USD per month [pricing.txt #2]", [(0.71234, {"source": "pricing.txt", "chunk_id": 2, "text": "..."})]


class FakeOpenAIError(openai.APIError):
    def __init__(self):
        Exception.__init__(self, "simulated outage")


def test_missing_key_is_rejected():
    assert client.post("/ask", json={"question": "hi"}).status_code == 401


def test_wrong_key_is_rejected():
    r = client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_empty_question_is_rejected():
    assert client.post("/ask", json={"question": ""}, headers=HEADERS).status_code == 422


def test_too_long_question_is_rejected():
    assert client.post("/ask", json={"question": "x" * 501}, headers=HEADERS).status_code == 422


def test_valid_request_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(api, "ask", fake_ask)
    r = client.post("/ask", json={"question": "How much is Growth?"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json() == {
        "answer": "899 USD per month [pricing.txt #2]",
        "sources": [{"source": "pricing.txt", "chunk_id": 2, "score": 0.712}],
    }


def test_openai_outage_returns_503(monkeypatch):
    def failing_ask(question):
        raise FakeOpenAIError()

    monkeypatch.setattr(api, "ask", failing_ask)
    r = client.post("/ask", json={"question": "How much is Growth?"}, headers=HEADERS)
    assert r.status_code == 503


def test_health_checks_the_database(monkeypatch):
    class FakePool:
        @contextmanager
        def connection(self):
            yield type("FakeConn", (), {"execute": lambda self, sql: None})()

    monkeypatch.setattr(api, "get_pool", lambda: FakePool())
    assert client.get("/health").json() == {"status": "ok"}
