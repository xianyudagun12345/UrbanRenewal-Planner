"""Small in-memory rate limiter for local product hardening.

This is intentionally dependency-free. It is not a replacement for Redis or an
API gateway in production, but it gives the FastAPI app a clear rate-limit seam
that can later be swapped for a distributed implementation.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: int = 0


class InMemoryRateLimiter:
    """Sliding-window request limiter keyed by IP/user/session."""

    def __init__(self, *, max_requests: int = 20, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> RateLimitResult:
        now = time.monotonic()
        window_start = now - self.window_seconds
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - hits[0])))
                return RateLimitResult(allowed=False, remaining=0, retry_after_seconds=retry_after)
            hits.append(now)
            return RateLimitResult(allowed=True, remaining=self.max_requests - len(hits))

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
