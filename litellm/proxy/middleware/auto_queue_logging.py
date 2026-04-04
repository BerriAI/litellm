from __future__ import annotations

import time
from typing import Any, Dict, MutableMapping, Optional

AUTOQ_METADATA_KEY = "autoq_metadata"
DEFAULT_AUTOQ_MAX_EVENTS = 20


def _ensure_autoq_metadata(
    request_data: MutableMapping[str, Any],
) -> Dict[str, Any]:
    autoq_metadata = request_data.get(AUTOQ_METADATA_KEY)
    if not isinstance(autoq_metadata, dict):
        autoq_metadata = {}
        request_data[AUTOQ_METADATA_KEY] = autoq_metadata

    if not isinstance(autoq_metadata.get("events"), list):
        autoq_metadata["events"] = []
    if not isinstance(autoq_metadata.get("summary"), dict):
        autoq_metadata["summary"] = {}

    return autoq_metadata


def append_autoq_event(
    request_data: MutableMapping[str, Any],
    *,
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    at_ms: Optional[int] = None,
    max_events: int = DEFAULT_AUTOQ_MAX_EVENTS,
) -> Dict[str, Any]:
    autoq_metadata = _ensure_autoq_metadata(request_data)
    autoq_events = autoq_metadata["events"]
    event_payload = payload if isinstance(payload, dict) else {}

    autoq_events.append(
        {
            "event": event,
            "at_ms": at_ms if at_ms is not None else int(time.time() * 1000),
            "payload": dict(event_payload),
        }
    )

    if max_events > 0 and len(autoq_events) > max_events:
        del autoq_events[:-max_events]

    return autoq_metadata


def finalize_autoq_summary(
    request_data: MutableMapping[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    autoq_metadata = _ensure_autoq_metadata(request_data)
    autoq_summary = autoq_metadata["summary"]

    if isinstance(summary, dict):
        autoq_summary.update(summary)

    return autoq_metadata
