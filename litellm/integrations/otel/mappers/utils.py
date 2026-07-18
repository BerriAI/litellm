"""Shared helpers for the attribute mappers.

Small, mapper-agnostic utilities — JSON serialization, message extraction, and
extractor-table application — pulled out of the individual mapper modules so
they live in one place.
"""

import json
from typing import Callable, Mapping, Sequence

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue
from litellm.integrations.otel.model.payloads import LLMCallSpanData


def drop_none(values: Mapping[str, AttrValue | None]) -> AttributeMap:
    """Return ``values`` with ``None``-valued entries removed."""
    return {k: v for k, v in values.items() if v is not None}


def collect(table: Mapping[str, Callable], source: object) -> AttributeMap:
    """Apply an extractor table to ``source``, dropping ``None`` results."""
    return drop_none({key: extract(source) for key, extract in table.items()})


def json_if(payload: Mapping[str, object]) -> str | None:
    """JSON-serialize ``payload`` only when it's non-empty; else ``None``."""
    return json.dumps(payload) if payload else None


def json_or_none(value: object) -> str | None:
    """JSON-serialize ``value`` (falling back to ``str``); ``None`` on failure."""
    try:
        return json.dumps(value, default=str)
    except Exception:
        return None


def stringify_message(message: object) -> str | None:
    """JSON-serialize a chat message dict; ``None`` if not a dict or on failure."""
    if not isinstance(message, dict):
        return None
    try:
        return json.dumps(message, default=str)
    except Exception:
        return None


def serialize_messages(messages: Sequence[object]) -> str | None:
    """Round-trip a sequence of message dicts through ``stringify_message``."""
    serialized = [json.loads(s) for s in (stringify_message(m) for m in messages) if s is not None]
    return json.dumps(serialized) if serialized else None


def message_content(message: object) -> str | None:
    """Extract the textual ``content`` from a chat message dict."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # multimodal: concatenate text parts only
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "".join(p for p in parts if isinstance(p, str)) or None
    return None


def output_messages(data: LLMCallSpanData) -> list:
    """The ``message`` payload of each response choice."""
    return [c.get("message") for c in data.choices_out if isinstance(c, dict)]
