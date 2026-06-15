"""Pydantic schemas for the UrbanRenewal Planner API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.urbanrenewal.agent.budget import AgentBudget, UserTier
from src.urbanrenewal.agent.plan import TaskPlan
from src.urbanrenewal.agent.report import PlanningReport

AudienceMode = Literal["professional", "public", "government"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    audience: AudienceMode = "professional"
    user_tier: UserTier = "anonymous"
    stream: bool = False


class ToolCallEvent(BaseModel):
    tool_name: str
    status: Literal["started", "completed", "failed"] = "completed"
    summary: str = ""
    data: dict[str, Any] | None = None


class AgentProgressEvent(BaseModel):
    type: Literal["progress", "tool", "token", "final", "error"]
    session_id: str
    message: str = ""
    payload: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[ToolCallEvent] = Field(default_factory=list)
    report: PlanningReport | None = None
    budget: AgentBudget | None = None
    task_plan: TaskPlan | None = None
    duration_ms: int
    cached: bool = False


class AsyncChatResponse(BaseModel):
    task_id: str
    session_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    status_url: str


class TaskStatusResponse(BaseModel):
    task_id: str
    session_id: str
    status: Literal["queued", "running", "succeeded", "failed", "not_found"]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    result: ChatResponse | None = None
    error: str = ""


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field("", max_length=1000)


class FeedbackResponse(BaseModel):
    ok: bool
    session_id: str
    received_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
