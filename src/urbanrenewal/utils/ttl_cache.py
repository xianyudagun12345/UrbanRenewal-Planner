"""Tiny thread-safe TTL cache used before introducing Redis."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """In-memory TTL cache with explicit get/set and no external dependencies."""

    def __init__(self, *, ttl_seconds: int = 300, max_items: int = 512) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: dict[str, CacheEntry[T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._items.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._items) >= self.max_items:
                self._evict_one(now)
            self._items[key] = CacheEntry(value=value, expires_at=now + self.ttl_seconds)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _evict_one(self, now: float) -> None:
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        if expired:
            self._items.pop(expired[0], None)
            return
        oldest_key = min(self._items, key=lambda key: self._items[key].expires_at)
        self._items.pop(oldest_key, None)
