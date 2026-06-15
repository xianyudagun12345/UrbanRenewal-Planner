from dataclasses import dataclass
from datetime import datetime

from fastapi.testclient import TestClient

from src.urbanrenewal.api.main import app
from src.urbanrenewal.api.schemas import ChatResponse
from src.urbanrenewal.api.store import InMemoryAPIStateStore


@dataclass
class DummyRecord:
    task_id: str = "task-demo"
    session_id: str = "session-demo"
    status: str = "queued"
    created_at: datetime = datetime(2026, 1, 1)
    updated_at: datetime = datetime(2026, 1, 1)
    result: ChatResponse | None = None
    error: str = ""


def test_async_chat_submit_and_status(monkeypatch):
    from src.urbanrenewal.api import main as api_main

    record = DummyRecord()
    monkeypatch.setattr(api_main._TASK_MANAGER, "submit", lambda request, **kwargs: record)
    monkeypatch.setattr(api_main._TASK_MANAGER, "get", lambda task_id: record if task_id == "task-demo" else None)
    monkeypatch.setattr(api_main, "_STATE_STORE", InMemoryAPIStateStore())
    api_main._RATE_LIMITER.reset()

    client = TestClient(app)
    payload = {"question": "请分析鞍山新村周边800米的老年友好问题", "session_id": "session-demo"}
    submit = client.post("/api/v1/chat/async", json=payload)
    assert submit.status_code == 200
    assert submit.json()["task_id"] == "task-demo"

    status = client.get("/api/v1/chat/tasks/task-demo")
    assert status.status_code == 200
    assert status.json()["status"] == "queued"

    missing = client.get("/api/v1/chat/tasks/missing")
    assert missing.status_code == 200
    assert missing.json()["status"] == "not_found"
