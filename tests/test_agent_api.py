import uuid

import openai
from fastapi.testclient import TestClient

import agent_service
import api

client = TestClient(api.app)       # no `with`: lifespan (DB, OpenAI warm-up) doesn't run in tests
HEADERS = {"X-API-Key": "test-api-key"}


class FakeOpenAIError(openai.APIError):
    def __init__(self):
        Exception.__init__(self, "simulated outage")


def fake_run_agent(calls):
    def _run(message, thread_id):
        calls.append(thread_id)
        return "Your bill is $1,199.", [{"name": "calculate_bill", "args": {"plan": "growth", "extra_minutes": 900}}]
    return _run


def test_agent_requires_api_key():
    assert client.post("/agent", json={"message": "hi"}).status_code == 401


def test_empty_message_is_rejected():
    assert client.post("/agent", json={"message": ""}, headers=HEADERS).status_code == 422


def test_thread_id_must_be_a_uuid():
    r = client.post("/agent", json={"message": "hi", "thread_id": "customer-42"}, headers=HEADERS)
    assert r.status_code == 422


def test_new_conversation_gets_a_thread_id(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "run_agent", fake_run_agent(calls))
    r = client.post("/agent", json={"message": "Growth with 900 extra minutes?"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Your bill is $1,199."
    assert body["tool_calls"][0]["name"] == "calculate_bill"
    uuid.UUID(body["thread_id"])                                   # a real UUID was generated


def test_thread_is_reused_and_scoped_to_the_api_key(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "run_agent", fake_run_agent(calls))
    thread = str(uuid.uuid4())
    r = client.post("/agent", json={"message": "And Scale?", "thread_id": thread}, headers=HEADERS)
    assert r.json()["thread_id"] == thread
    assert calls == [f"{api.KEY_ID}:{thread}"]                     # stored under this key's namespace


def test_openai_outage_returns_503(monkeypatch):
    def failing(message, thread_id):
        raise FakeOpenAIError()
    monkeypatch.setattr(api, "run_agent", failing)
    assert client.post("/agent", json={"message": "hi"}, headers=HEADERS).status_code == 503


def test_calculate_bill_tool_reports_bad_input_to_the_model():
    assert agent_service.calculate_bill.invoke({"plan": "growth", "extra_minutes": -5}).startswith("ERROR:")
    assert '"total_usd_per_month": 1199' in agent_service.calculate_bill.invoke({"plan": "growth", "extra_minutes": 900})
