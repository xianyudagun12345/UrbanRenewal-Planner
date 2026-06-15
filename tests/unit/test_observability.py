import logging

from src.urbanrenewal.api.observability import log_event


def test_log_event_returns_structured_payload(caplog):
    caplog.set_level(logging.INFO, logger="urbanrenewal.api")
    payload = log_event("unit.event", session_id="demo", duration_ms=12)

    assert payload["event"] == "unit.event"
    assert payload["session_id"] == "demo"
    assert payload["duration_ms"] == 12
    assert "timestamp" in payload
    assert "unit.event" in caplog.text
