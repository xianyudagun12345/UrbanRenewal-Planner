from datetime import datetime

from src.urbanrenewal.api.schemas import ChatResponse
from src.urbanrenewal.api.task_models import AgentTaskRecord
from src.urbanrenewal.api.task_store import InMemoryTaskStore, SQLiteTaskStore


def _record(status: str = "queued") -> AgentTaskRecord:
    return AgentTaskRecord(
        task_id="task-1",
        session_id="session-1",
        status=status,
        created_at=datetime(2026, 1, 1, 10, 0, 0),
        updated_at=datetime(2026, 1, 1, 10, 1, 0),
    )


def test_in_memory_task_store_saves_and_counts_records():
    store = InMemoryTaskStore()
    store.save(_record("queued"))
    store.save(AgentTaskRecord(task_id="task-2", session_id="session-2", status="failed"))

    assert store.get("task-1").session_id == "session-1"
    assert store.get("missing") is None
    assert store.snapshot() == {
        "total_tasks": 2,
        "queued": 1,
        "running": 0,
        "succeeded": 0,
        "failed": 1,
    }


def test_sqlite_task_store_persists_result(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    record = _record("succeeded")
    record.result = ChatResponse(
        session_id="session-1",
        answer="done",
        duration_ms=123,
    )
    store.save(record)

    reloaded = store.get("task-1")

    assert reloaded is not None
    assert reloaded.status == "succeeded"
    assert reloaded.result is not None
    assert reloaded.result.answer == "done"
    assert store.snapshot()["succeeded"] == 1


def test_sqlite_task_store_updates_existing_record(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    record = _record("queued")
    store.save(record)
    record.status = "running"
    record.updated_at = datetime(2026, 1, 1, 10, 2, 0)
    store.save(record)

    reloaded = store.get("task-1")

    assert reloaded is not None
    assert reloaded.status == "running"
    assert store.snapshot()["total_tasks"] == 1
