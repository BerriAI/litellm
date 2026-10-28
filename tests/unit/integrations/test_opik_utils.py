"""Unit tests for the native Opik integration's UUIDv7 id generation."""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

from litellm.integrations.opik.utils import create_uuid7, get_traces_and_spans_from_payload


def _timestamp_ms(uuid_str: str) -> int:
    """Return the unix-ms timestamp encoded in a UUIDv7's top 48 bits."""
    return uuid.UUID(uuid_str).int >> 80


def test_create_uuid7_is_valid_version_7_uuid():
    parsed = uuid.UUID(create_uuid7())
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122


def test_create_uuid7_encodes_timestamp_in_milliseconds():
    fixed = datetime(2026, 6, 24, 10, 0, 0, tzinfo=timezone.utc)

    with patch(
        "litellm.integrations.opik.utils.time.time", return_value=fixed.timestamp()
    ):
        value = create_uuid7()

    assert _timestamp_ms(value) == int(fixed.timestamp() * 1000)


def test_queued_opik_payloads_are_split_into_traces_and_spans_without_their_null_fields():
    traces, spans = get_traces_and_spans_from_payload(
        [
            {"id": "trace-1", "name": "chat.completion", "thread_id": None, "input": {"kept": None}},
            {"id": "span-1", "type": "llm", "parent_span_id": None, "tags": [], "total_cost": 0},
            {"id": "span-2", "type": None},
        ]
    )

    assert (traces, spans) == (
        [{"id": "trace-1", "name": "chat.completion", "input": {"kept": None}}],
        [{"id": "span-1", "type": "llm", "tags": [], "total_cost": 0}, {"id": "span-2"}],
    )


def test_an_empty_opik_queue_has_no_traces_and_no_spans():
    assert get_traces_and_spans_from_payload([]) == ([], [])
