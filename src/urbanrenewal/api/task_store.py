"""Persistence stores for asynchronous agent task records."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from src.urbanrenewal.api.schemas import ChatResponse
from src.urbanrenewal.api.task_models import AgentTaskRecord, TaskStatus


class TaskStore(Protocol):
    def save(self, record: AgentTaskRecord) -> None: ...

    def get(self, task_id: str) -> AgentTaskRecord | None: ...

    def snapshot(self) -> dict[str, Any]: ...


class InMemoryTaskStore:
    """Thread-safe in-memory task store for unit tests and local fallback."""

    def __init__(self) -> None:
        self._records: dict[str, AgentTaskRecord] = {}
        self._lock = Lock()

    def save(self, record: AgentTaskRecord) -> None:
        with self._lock:
            self._records[record.task_id] = record

    def get(self, task_id: str) -> AgentTaskRecord | None:
        with self._lock:
            return self._records.get(task_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
        return _snapshot(records)


class SQLiteTaskStore:
    """SQLite-backed task store for local persistent async task status."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def save(self, record: AgentTaskRecord) -> None:
        result_json = ""
        if record.result is not None:
            result_json = json.dumps(record.result.model_dump(mode="json"), ensure_ascii=False, default=str)
        with self._connect() as conn, self._lock:
            conn.execute(
                """
                INSERT INTO agent_tasks(
                    task_id, session_id, status, created_at, updated_at, result_json, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    result_json=excluded.result_json,
                    error=excluded.error
                """,
                (
                    record.task_id,
                    record.session_id,
                    record.status,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    result_json,
                    record.error,
                ),
            )

    def get(self, task_id: str) -> AgentTaskRecord | None:
        with self._connect() as conn, self._lock:
            row = conn.execute(
                """
                SELECT task_id, session_id, status, created_at, updated_at, result_json, error
                FROM agent_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        result = ChatResponse.model_validate(json.loads(row[5])) if row[5] else None
        return AgentTaskRecord(
            task_id=row[0],
            session_id=row[1],
            status=_as_task_status(row[2]),
            created_at=_parse_datetime(row[3]),
            updated_at=_parse_datetime(row[4]),
            result=result,
            error=row[6] or "",
        )

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as conn, self._lock:
            rows = conn.execute("SELECT status, COUNT(*) FROM agent_tasks GROUP BY status").fetchall()
            total = conn.execute("SELECT COUNT(*) FROM agent_tasks").fetchone()[0]
        counts = {status: count for status, count in rows}
        return {
            "total_tasks": int(total),
            "queued": int(counts.get("queued", 0)),
            "running": int(counts.get("running", 0)),
            "succeeded": int(counts.get("succeeded", 0)),
            "failed": int(counts.get("failed", 0)),
        }

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    error TEXT NOT NULL
                )
                """
            )


def _snapshot(records: list[AgentTaskRecord]) -> dict[str, Any]:
    return {
        "total_tasks": len(records),
        "queued": sum(1 for r in records if r.status == "queued"),
        "running": sum(1 for r in records if r.status == "running"),
        "succeeded": sum(1 for r in records if r.status == "succeeded"),
        "failed": sum(1 for r in records if r.status == "failed"),
    }


def _parse_datetime(value: str) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(value)


def _as_task_status(value: str) -> TaskStatus:
    if value in {"queued", "running", "succeeded", "failed"}:
        return value  # type: ignore[return-value]
    return "failed"


def default_sqlite_task_store() -> SQLiteTaskStore:
    root = Path(__file__).resolve().parents[3]
    db_path = os.getenv("URBANRENEWAL_API_TASK_DB") or str(root / "outputs" / "api_tasks.sqlite3")
    return SQLiteTaskStore(db_path)
