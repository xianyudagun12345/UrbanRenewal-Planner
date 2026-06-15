from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.urbanrenewal.agent.checkpoint import default_checkpointer


def test_default_checkpointer_uses_sqlite(monkeypatch, tmp_path):
    db_path = tmp_path / "agent_checkpoints.sqlite3"
    monkeypatch.setenv("URBANRENEWAL_AGENT_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("URBANRENEWAL_AGENT_CHECKPOINT_DB", str(db_path))
    default_checkpointer.cache_clear()

    saver = default_checkpointer()

    assert isinstance(saver, SqliteSaver)
    assert db_path.exists()
    default_checkpointer.cache_clear()


def test_default_checkpointer_can_use_memory(monkeypatch):
    monkeypatch.setenv("URBANRENEWAL_AGENT_CHECKPOINTER", "memory")
    default_checkpointer.cache_clear()

    saver = default_checkpointer()

    assert isinstance(saver, MemorySaver)
    default_checkpointer.cache_clear()
