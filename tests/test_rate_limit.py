from fastapi.testclient import TestClient

import api
import rate_limit
from rate_limit import RateLimiter

client = TestClient(api.app)
HEADERS = {"X-API-Key": "test-api-key"}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(limit=3, clock=FakeClock())
    assert [limiter.check("k") for _ in range(3)] == [0, 0, 0]
    assert limiter.check("k") > 0


def test_window_slides_so_old_calls_stop_counting():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60, clock=clock)
    limiter.check("k")
    limiter.check("k")
    assert limiter.check("k") > 0
    clock.now += 61                                   # a minute later the old calls have expired
    assert limiter.check("k") == 0


def test_keys_are_counted_separately():
    limiter = RateLimiter(limit=1, clock=FakeClock())
    assert limiter.check("a") == 0
    assert limiter.check("b") == 0


def test_api_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(rate_limit, "limiter", RateLimiter(limit=2, clock=FakeClock()))
    monkeypatch.setattr(api, "ask", lambda q: ("ok", []))
    codes = [client.post("/ask", json={"question": "hi"}, headers=HEADERS).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/ask", json={"question": "hi"}, headers=HEADERS)
    assert int(r.headers["Retry-After"]) >= 1


def test_bad_keys_do_not_use_up_the_limit(monkeypatch):
    monkeypatch.setattr(rate_limit, "limiter", RateLimiter(limit=1, clock=FakeClock()))
    monkeypatch.setattr(api, "ask", lambda q: ("ok", []))
    for _ in range(5):
        client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
    assert client.post("/ask", json={"question": "hi"}, headers=HEADERS).status_code == 200
