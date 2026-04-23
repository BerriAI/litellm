"""Tests for FocusVantageDestination behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.vantage_destination import (
    FocusVantageDestination,
    VANTAGE_MAX_BYTES_PER_UPLOAD,
    VANTAGE_MAX_ROWS_PER_UPLOAD,
)

MOCK_TARGET = "litellm.integrations.focus.destinations.vantage_destination.get_async_httpx_client"


def _window(freq: str = "hourly", hour: int = 5) -> FocusTimeWindow:
    start = datetime(2024, 1, 2, hour, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    return FocusTimeWindow(start_time=start, end_time=end, frequency=freq)


def _config(**overrides: Any) -> dict[str, Any]:
    base = {
        "api_key": "test-api-key",
        "integration_token": "test-token-123",
    }
    base.update(overrides)
    return base


def test_should_require_api_key():
    with pytest.raises(ValueError, match="api_key"):
        FocusVantageDestination(
            prefix="exports",
            config={"integration_token": "tok"},
        )


def test_should_require_integration_token():
    with pytest.raises(ValueError, match="integration_token"):
        FocusVantageDestination(
            prefix="exports",
            config={"api_key": "key"},
        )


def test_should_initialize_with_valid_config():
    dest = FocusVantageDestination(prefix="exports", config=_config())
    assert dest.api_key == "test-api-key"
    assert dest.integration_token == "test-token-123"
    assert dest.base_url == "https://api.vantage.sh"


def test_should_use_custom_base_url():
    dest = FocusVantageDestination(
        prefix="exports",
        config=_config(base_url="https://custom.vantage.sh"),
    )
    assert dest.base_url == "https://custom.vantage.sh"


@pytest.mark.asyncio
async def test_should_skip_empty_content():
    dest = FocusVantageDestination(prefix="exports", config=_config())
    # Should not raise
    await dest.deliver(content=b"", time_window=_window(), filename="usage.csv")


@pytest.mark.asyncio
async def test_should_upload_csv_to_correct_url():
    dest = FocusVantageDestination(prefix="exports", config=_config())

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(
            content=b"header\nrow1\n",
            time_window=_window(),
            filename="usage.csv",
        )

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "test-token-123" in call_args[0][0]
    assert "costs.csv" in call_args[0][0]
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_should_batch_large_content():
    dest = FocusVantageDestination(prefix="exports", config=_config())

    # Create content larger than 2 MB — use supported column names so
    # _strip_unsupported_columns does not remove them.
    header = b"ChargeCategory,ChargePeriodStart,BilledCost"
    row = b"a" * 100 + b"," + b"b" * 100 + b"," + b"c" * 100
    num_rows = (VANTAGE_MAX_BYTES_PER_UPLOAD // len(row)) + 100
    large_content = header + b"\n" + b"\n".join([row] * num_rows) + b"\n"

    assert len(large_content) > VANTAGE_MAX_BYTES_PER_UPLOAD

    upload_calls: List[bytes] = []

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None

    mock_client = MagicMock()

    async def capture_post(url, **kwargs):
        files = kwargs.get("files", {})
        if "csv" in files:
            upload_calls.append(files["csv"][1])
        return mock_response

    mock_client.post = capture_post

    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(
            content=large_content,
            time_window=_window(),
            filename="usage.csv",
        )

    # Should have made multiple uploads
    assert len(upload_calls) > 1
    # Each upload should be within limits
    for chunk in upload_calls:
        assert len(chunk) <= VANTAGE_MAX_BYTES_PER_UPLOAD


@pytest.mark.asyncio
async def test_should_batch_by_row_count():
    """Verify batching triggers when row count exceeds 10K even if under 2 MB."""
    dest = FocusVantageDestination(prefix="exports", config=_config())

    header = b"ChargeCategory"
    # Short rows so total size stays well under 2 MB
    row = b"x"
    num_rows = VANTAGE_MAX_ROWS_PER_UPLOAD + 500
    content = header + b"\n" + b"\n".join([row] * num_rows) + b"\n"

    # Confirm content is under 2 MB but over 10K rows
    assert len(content) < VANTAGE_MAX_BYTES_PER_UPLOAD
    assert num_rows > VANTAGE_MAX_ROWS_PER_UPLOAD

    upload_calls: List[bytes] = []

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None

    mock_client = MagicMock()

    async def capture_post(url, **kwargs):
        files = kwargs.get("files", {})
        if "csv" in files:
            upload_calls.append(files["csv"][1])
        return mock_response

    mock_client.post = capture_post

    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(
            content=content,
            time_window=_window(),
            filename="usage.csv",
        )

    # Should have made at least 2 uploads due to row count
    assert len(upload_calls) >= 2
    # Each batch should have at most 10K data rows (header + data rows + trailing newline)
    for chunk in upload_calls:
        lines = chunk.split(b"\n")
        data_lines = [line for line in lines[1:] if line.strip()]
        assert len(data_lines) <= VANTAGE_MAX_ROWS_PER_UPLOAD
