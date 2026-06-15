"""FastAPI application for the autonomous UrbanRenewal Planner Agent."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.urbanrenewal.agent.autonomous import run_autonomous, stream_autonomous
from src.urbanrenewal.agent.budget import budget_for_tier
from src.urbanrenewal.api.observability import log_event
from src.urbanrenewal.api.rate_limit import InMemoryRateLimiter
from src.urbanrenewal.api.schemas import (
    AgentProgressEvent,
    AsyncChatResponse,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    TaskStatusResponse,
    ToolCallEvent,
)
from src.urbanrenewal.api.store import default_sqlite_store
from src.urbanrenewal.api.tasks import AgentTaskManager

app = FastAPI(
    title="UrbanRenewal Planner API",
    version="0.1.0",
    description="Autonomous AI Agent API for Yangpu urban renewal planning.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATE_STORE = default_sqlite_store()
_RATE_LIMITER = InMemoryRateLimiter(max_requests=30, window_seconds=60)
_TASK_MANAGER = AgentTaskManager(max_workers=4)
_MAX_STORED_TURNS = 20
_MAX_CONTEXT_TURNS = 4


def _client_key(request: Request, session_id: str = "") -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    return f"{ip}:{session_id}"


def _enforce_rate_limit(request: Request, session_id: str = "") -> None:
    result = _RATE_LIMITER.check(_client_key(request, session_id))
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "请求过于频繁，请稍后重试。",
                "retry_after_seconds": result.retry_after_seconds,
            },
            headers={"Retry-After": str(result.retry_after_seconds)},
        )


def _tool_calls_from_events(events: list[dict[str, Any]]) -> list[ToolCallEvent]:
    return [
        ToolCallEvent(
            tool_name=event.get("tool_name", "unknown"),
            status=event.get("status", "completed"),
            summary=(event.get("summary") or "")[:240],
            data={"args": event.get("args", {})} if event.get("args") else None,
        )
        for event in events
    ]


def _load_session_payload(session_id: str) -> dict[str, Any]:
    payload = _STATE_STORE.get_session(session_id)
    return payload if isinstance(payload, dict) else {}


def _turns_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    turns = payload.get("turns", [])
    if isinstance(turns, list):
        cleaned = [
            {
                "question": str(turn.get("question", ""))[:1000],
                "answer": str(turn.get("answer", ""))[:2000],
            }
            for turn in turns
            if isinstance(turn, dict)
        ]
        if cleaned:
            return cleaned
    last_question = str(payload.get("last_question", "") or "")
    last_answer = str(payload.get("last_answer", "") or "")
    if last_question or last_answer:
        return [{"question": last_question[:1000], "answer": last_answer[:2000]}]
    return []


def _conversation_context_for_session(session_id: str) -> str:
    turns = _turns_from_payload(_load_session_payload(session_id))
    recent_turns = turns[-_MAX_CONTEXT_TURNS:]
    if not recent_turns:
        return ""
    lines = ["Recent persisted conversation turns:"]
    for index, turn in enumerate(recent_turns, start=1):
        lines.append(f"Turn {index} user: {turn['question']}")
        lines.append(f"Turn {index} assistant: {turn['answer']}")
    return "\n".join(lines)


def _save_chat_session(
    *,
    session_id: str,
    question: str,
    answer: str,
    tool_calls: list[ToolCallEvent],
    duration_ms: int,
) -> None:
    payload = _load_session_payload(session_id)
    turns = _turns_from_payload(payload)
    turns.append({"question": question, "answer": answer})
    turns = turns[-_MAX_STORED_TURNS:]
    _STATE_STORE.save_session(session_id, {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "last_question": question,
        "last_answer": answer,
        "turns": turns,
        "tool_calls": [event.model_dump() for event in tool_calls],
        "duration_ms": duration_ms,
    })


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="urbanrenewal-planner", version=app.version)


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request_body: ChatRequest, request: Request) -> ChatResponse:
    _enforce_rate_limit(request, request_body.session_id)
    started = time.perf_counter()
    budget = budget_for_tier(request_body.user_tier)
    try:
        conversation_context = _conversation_context_for_session(request_body.session_id)
        result = run_autonomous(
            request_body.question,
            thread_id=request_body.session_id,
            budget=budget,
            conversation_context=conversation_context,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        tool_calls = _tool_calls_from_events(result.tool_events)
        _save_chat_session(
            session_id=request_body.session_id,
            question=request_body.question,
            answer=result.answer,
            tool_calls=tool_calls,
            duration_ms=duration_ms,
        )
        log_event(
            "chat.completed",
            session_id=request_body.session_id,
            duration_ms=duration_ms,
            user_tier=request_body.user_tier,
            tool_count=len(tool_calls),
            clarification=result.task_plan.clarification.needed if result.task_plan else False,
        )
        return ChatResponse(
            session_id=request_body.session_id,
            answer=result.answer,
            tool_calls=tool_calls,
            report=result.report,
            budget=result.budget,
            task_plan=result.task_plan,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_event("chat.failed", session_id=request_body.session_id, duration_ms=duration_ms, error=str(exc))
        raise


@app.post("/api/v1/chat/async", response_model=AsyncChatResponse)
def chat_async(request_body: ChatRequest, request: Request) -> AsyncChatResponse:
    _enforce_rate_limit(request, request_body.session_id)
    conversation_context = _conversation_context_for_session(request_body.session_id)
    record = _TASK_MANAGER.submit(request_body, conversation_context=conversation_context)
    log_event("chat_async.submitted", session_id=request_body.session_id, task_id=record.task_id, user_tier=request_body.user_tier)
    return AsyncChatResponse(
        task_id=record.task_id,
        session_id=record.session_id,
        status=record.status,
        status_url=f"/api/v1/chat/tasks/{record.task_id}",
    )


@app.get("/api/v1/chat/tasks/{task_id}", response_model=TaskStatusResponse)
def get_chat_task(task_id: str) -> TaskStatusResponse:
    record = _TASK_MANAGER.get(task_id)
    if record is None:
        log_event("chat_task.not_found", task_id=task_id)
        return TaskStatusResponse(task_id=task_id, session_id="", status="not_found")
    log_event("chat_task.read", task_id=task_id, session_id=record.session_id, status=record.status)
    return TaskStatusResponse(
        task_id=record.task_id,
        session_id=record.session_id,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        result=record.result,
        error=record.error,
    )


@app.post("/api/v1/chat/stream")
def chat_stream(request_body: ChatRequest, request: Request) -> StreamingResponse:
    _enforce_rate_limit(request, request_body.session_id)
    budget = budget_for_tier(request_body.user_tier)
    conversation_context = _conversation_context_for_session(request_body.session_id)
    log_event("chat_stream.started", session_id=request_body.session_id, user_tier=request_body.user_tier)

    def _dump_event(event: AgentProgressEvent) -> str:
        return json.dumps(event.model_dump(mode="json"), ensure_ascii=False, default=str) + "\n"

    def _iter_events():
        yield _dump_event(AgentProgressEvent(
            type="progress",
            session_id=request_body.session_id,
            message="正在启动自主 Agent",
        ))
        try:
            last_event: dict[str, Any] | None = None
            for item in stream_autonomous(
                request_body.question,
                thread_id=request_body.session_id,
                budget=budget,
                conversation_context=conversation_context,
            ):
                last_event = item
                yield _dump_event(AgentProgressEvent(
                    type="progress",
                    session_id=request_body.session_id,
                    message="Agent 状态更新",
                    payload={"event": item["event"]},
                ))
            _STATE_STORE.save_session(request_body.session_id, {
                "session_id": request_body.session_id,
                "updated_at": datetime.now().isoformat(),
                "last_question": request_body.question,
                "last_stream_event": str(last_event) if last_event else "",
            })
            yield _dump_event(AgentProgressEvent(
                type="final",
                session_id=request_body.session_id,
                message="分析完成",
            ))
            log_event("chat_stream.completed", session_id=request_body.session_id)
        except Exception as exc:
            log_event("chat_stream.failed", session_id=request_body.session_id, error=str(exc))
            yield _dump_event(AgentProgressEvent(
                type="error",
                session_id=request_body.session_id,
                message=str(exc),
            ))

    return StreamingResponse(_iter_events(), media_type="application/x-ndjson")


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return _STATE_STORE.get_session(session_id) or {"session_id": session_id, "found": False}


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
def feedback(request_body: FeedbackRequest, request: Request) -> FeedbackResponse:
    _enforce_rate_limit(request, request_body.session_id)
    received_at = datetime.now()
    _STATE_STORE.add_feedback({**request_body.model_dump(), "received_at": received_at.isoformat()})
    log_event("feedback.received", session_id=request_body.session_id, rating=request_body.rating)
    return FeedbackResponse(ok=True, session_id=request_body.session_id, received_at=received_at)


@app.get("/api/v1/service/status")
def service_status() -> dict[str, Any]:
    return {
        "sessions": _STATE_STORE.count_sessions(),
        "feedback": _STATE_STORE.count_feedback(),
        "tasks": _TASK_MANAGER.snapshot(),
        "rate_limit": {
            "max_requests": _RATE_LIMITER.max_requests,
            "window_seconds": _RATE_LIMITER.window_seconds,
        },
    }
