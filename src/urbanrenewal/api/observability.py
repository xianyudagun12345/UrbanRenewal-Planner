"""Structured event logging helpers for API observability."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("urbanrenewal.api")


def log_event(event: str, **fields: Any) -> dict[str, Any]:
    """Log one structured event and return the serialized payload for tests."""
    payload = {
        "event": event,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))
    return payload
