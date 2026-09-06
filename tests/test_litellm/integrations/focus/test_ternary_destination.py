"""Tests for FocusTernaryDestination behavior."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm.integrations.focus.destinations.ternary_destination as td
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.ternary_destination import (
    TERNARY_UPLOAD_TIMEOUT_SECONDS,
    FocusTernaryDestination,
)

MOCK_TARGET = "litellm.integrations.focus.destinations.ternary_destination.get_async_httpx_client"


def _window(freq: str = "daily", hour: int = 5) -> FocusTimeWindow:
    start = datetime(2024, 1, 2, hour, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    return FocusTimeWindow(start_time=start, end_time=end, frequency=freq)


def _config(**overrides: Any) -> dict[str, Any]:
    base = {
        "api_key": "test-api-key",
        "connection_id": "conn-1234",
        "base_url": "https://ternary.test",
    }
    base.update(overrides)
    return base


def _capturing_client() -> tuple[MagicMock, list[dict[str, Any]]]:
    """Return a mock client plus a list that records each post() call."""
    calls: list[dict[str, Any]] = []

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None

    mock_client = MagicMock()

    async def capture_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return mock_response

    mock_client.post = capture_post
    return mock_client, calls


def _uploaded_part(call: dict[str, Any]) -> tuple[str, bytes]:
    field = call["files"]["csv"]
    return field[0], field[1]


def _rows(content: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(content.decode("utf-8"))))


def test_should_require_api_key():
    with pytest.raises(ValueError, match="api_key"):
        FocusTernaryDestination(prefix="exports", config={"connection_id": "c", "base_url": "u"})


def test_should_require_connection_id():
    with pytest.raises(ValueError, match="connection_id"):
        FocusTernaryDestination(prefix="exports", config={"api_key": "k", "base_url": "u"})


def test_should_require_base_url():
    with pytest.raises(ValueError, match="base_url"):
        FocusTernaryDestination(prefix="exports", config={"api_key": "k", "connection_id": "c"})


@pytest.mark.parametrize("bad_id", ["a/b", "..", "has space", "tab\tid"])
def test_should_reject_connection_id_that_could_reroute_the_path(bad_id):
    with pytest.raises(ValueError, match="connection_id"):
        FocusTernaryDestination(prefix="exports", config=_config(connection_id=bad_id))


def test_should_initialize_with_valid_config():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    assert dest.api_key == "test-api-key"
    assert dest.connection_id == "conn-1234"
    assert dest.base_url == "https://ternary.test"


def test_should_use_custom_base_url_and_strip_trailing_slash():
    dest = FocusTernaryDestination(prefix="exports", config=_config(base_url="http://localhost:8080/"))
    assert dest.base_url == "http://localhost:8080"


@pytest.mark.asyncio
async def test_should_skip_empty_content():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    with patch(MOCK_TARGET, side_effect=AssertionError("client initialized on empty content")):
        assert await dest.deliver(content=b"", time_window=_window(), filename="usage.csv") is None


@pytest.mark.asyncio
async def test_should_upload_to_correct_url_with_auth_and_upload_headers():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    mock_client, calls = _capturing_client()

    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=b"header\nrow1\n", time_window=_window(), filename="usage.csv")

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://ternary.test/external-cost-sources/v1/conn-1234/focus"
    assert call["headers"]["Authorization"] == "Bearer test-api-key"
    assert call["headers"]["X-Ternary-Chunk-Index"] == "0"
    assert call["headers"]["X-Ternary-Chunk-Total"] == "1"
    assert call["headers"]["X-Ternary-Upload-Id"]
    filename, body = _uploaded_part(call)
    assert filename == "usage.csv"
    assert body == b"header\nrow1\n"
    assert call["files"]["csv"][2] == "text/csv"
    assert call["timeout"] == TERNARY_UPLOAD_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_should_url_encode_the_connection_id():
    dest = FocusTernaryDestination(prefix="exports", config=_config(connection_id="conn+id~ok"))
    mock_client, calls = _capturing_client()
    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    assert calls[0]["url"].endswith("/external-cost-sources/v1/conn%2Bid~ok/focus")


@pytest.mark.asyncio
async def test_should_pass_tags_column_through_unstripped():
    """The Ternary sink must not drop/strip any columns (forwards Tags as-is)."""
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    mock_client, calls = _capturing_client()

    content = b'ServiceName,Tags,x_unknown\nfoo,"{""team_id"": ""t1""}",keepme\n'
    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=content, time_window=_window(), filename="usage.csv")

    _, body = _uploaded_part(calls[0])
    assert body == content


@pytest.mark.asyncio
async def test_should_chunk_by_row_count_with_stable_upload_id(monkeypatch):
    monkeypatch.setattr(td, "TERNARY_MAX_ROWS_PER_UPLOAD", 2)
    dest = FocusTernaryDestination(prefix="exports", config=_config())

    content = b"ServiceName\n" + b"\n".join([b"x"] * 5) + b"\n"
    mock_client, calls = _capturing_client()
    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=content, time_window=_window(), filename="usage.csv")

    assert len(calls) == 3
    total = "3"
    assert {c["headers"]["X-Ternary-Upload-Id"] for c in calls} == {calls[0]["headers"]["X-Ternary-Upload-Id"]}
    for i, call in enumerate(calls):
        assert call["headers"]["X-Ternary-Chunk-Index"] == str(i)
        assert call["headers"]["X-Ternary-Chunk-Total"] == total
        filename, body = _uploaded_part(call)
        assert filename == f"usage.csv.part{i + 1}"
        assert len(_rows(body)) - 1 <= 2


@pytest.mark.asyncio
async def test_should_chunk_by_bytes(monkeypatch):
    monkeypatch.setattr(td, "TERNARY_MAX_BYTES_PER_UPLOAD", 200)
    dest = FocusTernaryDestination(prefix="exports", config=_config())

    row = b"a" * 40 + b"," + b"b" * 40
    content = b"ServiceName,BilledCost\n" + b"\n".join([row] * 20) + b"\n"
    mock_client, calls = _capturing_client()
    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=content, time_window=_window(), filename="usage.csv")

    assert len(calls) > 1
    for call in calls:
        _, body = _uploaded_part(call)
        assert len(body) <= 200


@pytest.mark.asyncio
async def test_should_not_mangle_a_quoted_field_containing_a_newline(monkeypatch):
    monkeypatch.setattr(td, "TERNARY_MAX_ROWS_PER_UPLOAD", 1)
    dest = FocusTernaryDestination(prefix="exports", config=_config())

    content = b'ServiceName,Tags\nsvc1,"line1\nline2"\nsvc2,"{""k"":""v""}"\n'
    mock_client, calls = _capturing_client()
    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=content, time_window=_window(), filename="usage.csv")

    assert len(calls) == 2
    _, first = _uploaded_part(calls[0])
    rows = _rows(first)
    assert rows[0] == ["ServiceName", "Tags"]
    assert rows[1] == ["svc1", "line1\nline2"]


@pytest.mark.asyncio
async def test_should_raise_on_a_single_row_larger_than_the_byte_limit(monkeypatch):
    monkeypatch.setattr(td, "TERNARY_MAX_BYTES_PER_UPLOAD", 50)
    dest = FocusTernaryDestination(prefix="exports", config=_config())

    content = b"ServiceName\n" + b"z" * 200 + b"\n"
    mock_client, _ = _capturing_client()
    with patch(MOCK_TARGET, return_value=mock_client):
        with pytest.raises(ValueError, match="cannot be split"):
            await dest.deliver(content=content, time_window=_window(), filename="usage.csv")


@pytest.mark.asyncio
async def test_should_abort_on_first_chunk_failure(monkeypatch):
    monkeypatch.setattr(td, "TERNARY_MAX_ROWS_PER_UPLOAD", 1)
    dest = FocusTernaryDestination(prefix="exports", config=_config())

    content = b"ServiceName\n" + b"\n".join([b"x"] * 3) + b"\n"

    attempts: list[str] = []
    mock_client = MagicMock()

    async def failing_post(url, **kwargs):
        attempts.append(kwargs["headers"]["X-Ternary-Chunk-Index"])
        raise RuntimeError("boom")

    mock_client.post = failing_post

    with patch(MOCK_TARGET, return_value=mock_client):
        with pytest.raises(RuntimeError, match="boom"):
            await dest.deliver(content=content, time_window=_window(), filename="usage.csv")

    assert attempts == ["0"]


@pytest.mark.parametrize("url", ["http://api.ternary.app", "http://evil.example.com:8080", "ftp://ternary.test"])
def test_should_reject_non_https_base_url(url):
    with pytest.raises(ValueError, match="HTTPS"):
        FocusTernaryDestination(prefix="exports", config=_config(base_url=url))


@pytest.mark.parametrize("url", ["https://ternary.test", "http://localhost:8080", "http://127.0.0.1:8080"])
def test_should_accept_https_or_loopback_base_url(url):
    dest = FocusTernaryDestination(prefix="exports", config=_config(base_url=url))
    assert dest.base_url == url.rstrip("/")


@pytest.mark.asyncio
async def test_should_return_header_only_content_untouched(monkeypatch):
    monkeypatch.setattr(td, "TERNARY_MAX_BYTES_PER_UPLOAD", 10)
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    content = b"HeaderOnlyLineWithNoDataRows"
    mock_client, calls = _capturing_client()
    with patch(MOCK_TARGET, return_value=mock_client):
        await dest.deliver(content=content, time_window=_window(), filename="usage.csv")
    assert len(calls) == 1
    _, body = _uploaded_part(calls[0])
    assert body == content
