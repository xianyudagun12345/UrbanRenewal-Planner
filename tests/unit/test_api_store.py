from src.urbanrenewal.api.store import InMemoryAPIStateStore, SQLiteAPIStateStore


def test_in_memory_api_state_store():
    store = InMemoryAPIStateStore()
    store.save_session("s1", {"session_id": "s1", "updated_at": "now"})
    store.add_feedback({"session_id": "s1", "rating": 5, "received_at": "now"})

    assert store.get_session("s1")["session_id"] == "s1"
    assert store.count_sessions() == 1
    assert store.count_feedback() == 1


def test_sqlite_api_state_store_persists(tmp_path):
    db_path = tmp_path / "api_state.sqlite3"
    store = SQLiteAPIStateStore(db_path)
    store.save_session("s1", {"session_id": "s1", "updated_at": "now", "last_question": "q"})
    store.add_feedback({"session_id": "s1", "rating": 4, "received_at": "now"})

    reopened = SQLiteAPIStateStore(db_path)
    assert reopened.get_session("s1")["last_question"] == "q"
    assert reopened.count_sessions() == 1
    assert reopened.count_feedback() == 1
