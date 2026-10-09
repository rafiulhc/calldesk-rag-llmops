import json
from types import SimpleNamespace

import pytest

import ask
from request_log import LOG_FILE

CHUNK = {"source": "pricing.txt", "chunk_id": 2, "text": "Growth plan: 899 USD per month."}


class FakeGPT:
    """Stands in for client.responses: records calls, returns a fixed answer."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="899 USD per month [pricing.txt #2]", model="fake-model",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def fake_search(score):
    def _search(question, top_k):
        return [(score, CHUNK)], 5
    return _search


@pytest.fixture
def gpt(monkeypatch):
    fake = FakeGPT()
    monkeypatch.setattr(ask, "client", SimpleNamespace(responses=fake))
    return fake


def last_log():
    return json.loads(LOG_FILE.read_text().splitlines()[-1])


def test_low_score_refuses_without_calling_gpt(monkeypatch, gpt):
    monkeypatch.setattr(ask, "search", fake_search(0.1))
    answer, results = ask.ask("What is the capital of France?")
    assert answer == ask.REFUSAL_ANSWER
    assert results == []
    assert gpt.calls == []           # the important part: no money spent


def test_relevant_chunk_is_sent_to_gpt(monkeypatch, gpt):
    monkeypatch.setattr(ask, "search", fake_search(0.7))
    answer, results = ask.ask("How much is Growth?")
    assert answer.startswith("899")
    assert len(gpt.calls) == 1
    assert "Growth plan: 899 USD" in gpt.calls[0]["input"]
    assert "[pricing.txt #2]" in gpt.calls[0]["input"]


def test_every_request_is_logged(monkeypatch, gpt):
    monkeypatch.setattr(ask, "search", fake_search(0.7))
    ask.ask("How much is Growth?")
    log = last_log()
    assert log["question"] == "How much is Growth?"
    assert log["input_tokens"] == 100
    assert log["retrieved"][0]["source"] == "pricing.txt"
    assert "total_ms" in log


def test_errors_are_logged_and_reraised(monkeypatch, gpt):
    def broken_search(question, top_k):
        raise RuntimeError("database down")

    monkeypatch.setattr(ask, "search", broken_search)
    with pytest.raises(RuntimeError):
        ask.ask("Anything")
    assert "database down" in last_log()["error"]
