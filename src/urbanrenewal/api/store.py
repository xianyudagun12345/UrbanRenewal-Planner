"""Session and feedback stores for API state.

The app starts with SQLite because it is dependency-free and persistent. The
interface is intentionally small so production can later replace it with
PostgreSQL or Redis without changing API handlers.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


class APIStateStore(Protocol):
    def save_session(self, session_id: str, payload: dict[str, Any]) -> None: ...

    def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    def count_sessions(self) -> int: ...

    def add_feedback(self, payload: dict[str, Any]) -> None: ...

    def count_feedback(self) -> int: ...


class InMemoryAPIStateStore:
    """Test-friendly in-memory store implementing the production interface."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []

    def save_session(self, session_id: str, payload: dict[str, Any]) -> None:
        self.sessions[session_id] = payload

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    def count_sessions(self) -> int:
        return len(self.sessions)

    def add_feedback(self, payload: dict[str, Any]) -> None:
        self.feedback.append(payload)

    def count_feedback(self) -> int:
        return len(self.feedback)


class SQLiteAPIStateStore:
    """SQLite-backed API state store for local persistence."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def save_session(self, session_id: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        updated_at = str(payload.get("updated_at", ""))
        with self._connect() as conn, self._lock:
            conn.execute(
                """
                INSERT INTO sessions(session_id, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (session_id, updated_at, encoded),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, self._lock:
            row = conn.execute("SELECT payload_json FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def count_sessions(self) -> int:
        with self._connect() as conn, self._lock:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0])

    def add_feedback(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        session_id = str(payload.get("session_id", ""))
        received_at = str(payload.get("received_at", ""))
        rating = int(payload.get("rating", 0) or 0)
        with self._connect() as conn, self._lock:
            conn.execute(
                """
                INSERT INTO feedback(session_id, rating, received_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, rating, received_at, encoded),
            )

    def count_feedback(self) -> int:
        with self._connect() as conn, self._lock:
            row = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()
        return int(row[0])

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    received_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )


def default_sqlite_store() -> SQLiteAPIStateStore:
    root = Path(__file__).resolve().parents[3]
    db_path = os.getenv("URBANRENEWAL_API_STATE_DB") or str(root / "outputs" / "api_state.sqlite3")
    return SQLiteAPIStateStore(db_path)
