"""Shared error-detection helpers."""

import json
from typing import Any

_STREAM_REQUIRED_TEXT = "stream must be set to true"


def _contains_stream_required_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)
    if isinstance(value, str):
        lowered = value.lower()
        if _STREAM_REQUIRED_TEXT in lowered:
            return True
        try:
            parsed = json.loads(value)
        except Exception:
            return False
        return _contains_stream_required_text(parsed)
    if isinstance(value, dict):
        for key in ("detail", "message", "error"):
            if key in value and _contains_stream_required_text(value[key]):
                return True
        return any(_contains_stream_required_text(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_stream_required_text(v) for v in value)
    return False


def is_stream_required_error(err: Exception) -> bool:
    for attr in ("body", "message", "text"):
        if _contains_stream_required_text(getattr(err, attr, None)):
            return True
    response = getattr(err, "response", None)
    if response is not None:
        try:
            if _contains_stream_required_text(getattr(response, "text", None)):
                return True
        except Exception:
            return False
    return _contains_stream_required_text(str(err))
