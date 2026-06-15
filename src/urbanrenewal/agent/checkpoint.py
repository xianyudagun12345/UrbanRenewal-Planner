"""Checkpointer factory for autonomous agent conversation state."""

from __future__ import annotations

import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

CheckpointerBackend = Literal["sqlite", "memory"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _checkpoint_backend() -> CheckpointerBackend:
    value = os.getenv("URBANRENEWAL_AGENT_CHECKPOINTER", "sqlite").strip().lower()
    if value == "memory":
        return "memory"
    return "sqlite"


def _checkpoint_db_path() -> Path:
    configured = os.getenv("URBANRENEWAL_AGENT_CHECKPOINT_DB", "").strip()
    if configured:
        return Path(configured)
    return _project_root() / "outputs" / "agent_checkpoints.sqlite3"


@lru_cache(maxsize=1)
def default_checkpointer() -> Any:
    """Return the process-wide LangGraph checkpointer.

    SQLite gives the local API a persistent conversation state across process
    restarts. It is still a lightweight deployment choice; high-concurrency
    production should move this factory to Redis/PostgreSQL.
    """
    if _checkpoint_backend() == "memory":
        return MemorySaver()

    db_path = _checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


__all__ = ["CheckpointerBackend", "default_checkpointer"]
