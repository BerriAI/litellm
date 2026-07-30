"""Unit tests for the native Opik integration's UUIDv7 id generation."""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

from litellm.integrations.opik.utils import create_uuid7


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
