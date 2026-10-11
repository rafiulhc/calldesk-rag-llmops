"""A sliding-window rate limiter per API key, so a leaked or abused key can't run up the OpenAI bill.

In-process: correct for one container. With several containers, each one counts separately,
so the real limit becomes N x the setting; a shared store (Redis/ElastiCache) would fix that.
"""
import os
import threading
import time
from collections import defaultdict, deque

LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
WINDOW_SECONDS = 60


class RateLimiter:
    def __init__(self, limit, window=WINDOW_SECONDS, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.calls = defaultdict(deque)
        self.lock = threading.Lock()    # FastAPI runs sync endpoints in a thread pool

    def check(self, key):
        """Record a call. Returns 0 if allowed, otherwise the seconds to wait before retrying."""
        now = self.clock()
        with self.lock:
            calls = self.calls[key]
            while calls and calls[0] <= now - self.window:
                calls.popleft()                 # forget calls older than the window
            if len(calls) >= self.limit:
                return max(1, int(calls[0] + self.window - now) + 1)
            calls.append(now)
            return 0


limiter = RateLimiter(LIMIT_PER_MINUTE)
