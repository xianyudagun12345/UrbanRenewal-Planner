from fastapi.testclient import TestClient

from src.urbanrenewal.agent.autonomous import AutonomousRunResult
from src.urbanrenewal.api.main import app
from src.urbanrenewal.api.store import InMemoryAPIStateStore


def test_chat_endpoint_injects_persisted_session_context(monkeypatch):
    from src.urbanrenewal.api import main as api_main

    captured: dict[str, str] = {}
    api_main._STATE_STORE = InMemoryAPIStateStore()
    api_main._RATE_LIMITER.reset()
    api_main._STATE_STORE.save_session(
        "session-context",
        {
            "session_id": "session-context",
            "updated_at": "2026-01-01T00:00:00",
            "turns": [
                {
                    "question": "请分析鞍山新村周边800米的老年友好问题",
                    "answer": "已确认分析地点为鞍山新村，半径为800米。",
                }
            ],
        },
    )

    def fake_run_autonomous(question, *, thread_id=None, budget=None, conversation_context="", **kwargs):
        captured["question"] = question
        captured["thread_id"] = thread_id
        captured["conversation_context"] = conversation_context
        return AutonomousRunResult(session_id=thread_id, answer="ok")

    monkeypatch.setattr(api_main, "run_autonomous", fake_run_autonomous)

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "session-context", "question": "那附近有没有适合老人休息的空间？"},
    )

    assert response.status_code == 200
    assert captured["thread_id"] == "session-context"
    assert "鞍山新村" in captured["conversation_context"]
    assert "800米" in captured["conversation_context"]
    saved = api_main._STATE_STORE.get_session("session-context")
    assert len(saved["turns"]) == 2
    assert saved["last_answer"] == "ok"


def test_conversation_context_caps_recent_turns():
    from src.urbanrenewal.api import main as api_main

    api_main._STATE_STORE = InMemoryAPIStateStore()
    api_main._STATE_STORE.save_session(
        "session-cap",
        {
            "session_id": "session-cap",
            "updated_at": "2026-01-01T00:00:00",
            "turns": [
                {"question": f"q{i}", "answer": f"a{i}"}
                for i in range(1, 7)
            ],
        },
    )

    context = api_main._conversation_context_for_session("session-cap")

    assert "q1" not in context
    assert "q2" not in context
    assert "q3" in context
    assert "q6" in context
