from fastapi.testclient import TestClient

from src.urbanrenewal.api.main import app
from src.urbanrenewal.api.store import InMemoryAPIStateStore


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_service_status_endpoint():
    from src.urbanrenewal.api import main as api_main

    api_main._STATE_STORE = InMemoryAPIStateStore()
    client = TestClient(app)
    response = client.get("/api/v1/service/status")
    assert response.status_code == 200
    body = response.json()
    assert "tasks" in body
    assert "rate_limit" in body


def test_feedback_persists_to_store():
    from src.urbanrenewal.api import main as api_main

    api_main._STATE_STORE = InMemoryAPIStateStore()
    api_main._RATE_LIMITER.reset()
    client = TestClient(app)
    response = client.post("/api/v1/feedback", json={"session_id": "s1", "rating": 5, "comment": "ok"})

    assert response.status_code == 200
    assert api_main._STATE_STORE.count_feedback() == 1
