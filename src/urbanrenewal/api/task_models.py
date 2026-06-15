"""Shared models for asynchronous agent task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from src.urbanrenewal.api.schemas import ChatResponse

TaskStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class AgentTaskRecord:
    task_id: str
    session_id: str
    status: TaskStatus = "queued"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result: ChatResponse | None = None
    error: str = ""
