"""Background task manager for long-running agent requests."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.urbanrenewal.agent.autonomous import run_autonomous
from src.urbanrenewal.agent.budget import budget_for_tier
from src.urbanrenewal.api.observability import log_event
from src.urbanrenewal.api.schemas import ChatRequest, ChatResponse, ToolCallEvent
from src.urbanrenewal.api.task_models import AgentTaskRecord, TaskStatus
from src.urbanrenewal.api.task_store import TaskStore, default_sqlite_task_store


class AgentTaskManager:
    """Thread-pool backed task manager.

    SQLite persistence keeps status records queryable across API process
    restarts. Production can later replace the store/executor with Celery/RQ
    and Redis while preserving this submit/get surface.
    """

    def __init__(self, *, max_workers: int = 4, task_store: TaskStore | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-task")
        self._store = task_store or default_sqlite_task_store()
        self._futures: dict[str, Future[None]] = {}

    def submit(self, request: ChatRequest, *, conversation_context: str = "") -> AgentTaskRecord:
        task_id = str(uuid4())
        record = AgentTaskRecord(task_id=task_id, session_id=request.session_id)
        self._store.save(record)
        self._futures[task_id] = self._executor.submit(self._run, task_id, request, conversation_context)
        log_event("agent_task.queued", task_id=task_id, session_id=request.session_id, user_tier=request.user_tier)
        return record

    def get(self, task_id: str) -> AgentTaskRecord | None:
        return self._store.get(task_id)

    def snapshot(self) -> dict[str, Any]:
        return self._store.snapshot()

    def _run(self, task_id: str, request: ChatRequest, conversation_context: str = "") -> None:
        record = self._store.get(task_id)
        if record is None:
            log_event("agent_task.missing_record", task_id=task_id, session_id=request.session_id)
            return
        record.status = "running"
        record.updated_at = datetime.now()
        self._store.save(record)
        log_event("agent_task.started", task_id=task_id, session_id=request.session_id)
        started = time.perf_counter()
        try:
            budget = budget_for_tier(request.user_tier)
            result = run_autonomous(
                request.question,
                thread_id=request.session_id,
                budget=budget,
                conversation_context=conversation_context,
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            tool_calls = [
                ToolCallEvent(
                    tool_name=event.get("tool_name", "unknown"),
                    status=event.get("status", "completed"),
                    summary=(event.get("summary") or "")[:240],
                    data={"args": event.get("args", {})} if event.get("args") else None,
                )
                for event in result.tool_events
            ]
            record.result = ChatResponse(
                session_id=request.session_id,
                answer=result.answer,
                tool_calls=tool_calls,
                report=result.report,
                budget=result.budget,
                task_plan=result.task_plan,
                duration_ms=duration_ms,
            )
            record.status = "succeeded"
            log_event(
                "agent_task.succeeded",
                task_id=task_id,
                session_id=request.session_id,
                duration_ms=duration_ms,
                tool_count=len(tool_calls),
            )
        except Exception as exc:
            record.error = str(exc)
            record.status = "failed"
            log_event("agent_task.failed", task_id=task_id, session_id=request.session_id, error=str(exc))
        finally:
            record.updated_at = datetime.now()
            self._store.save(record)


__all__ = ["AgentTaskManager", "AgentTaskRecord", "TaskStatus"]
