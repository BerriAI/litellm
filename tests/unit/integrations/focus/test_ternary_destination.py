"""Tests for FocusTernaryDestination behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.ternary_destination import (
    TERNARY_UPLOAD_TIMEOUT_SECONDS,
    FocusTernaryDestination,
    _NoRedirectHTTPHandler,
)

MOCK_TARGET = "litellm.integrations.focus.destinations.ternary_destination._NoRedirectHTTPHandler"


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


_SIGNED_URL = "https://storage.googleapis.com/ter-ecs-t-us/days/cloud-1/2024-01-02.csv?X-Goog-Signature=abc"
_SIGNED_HEADERS = {
    "Content-Type": "text/csv",
    "x-goog-meta-ecs-provider": "litellm",
    "x-goog-meta-ecs-connection-id": "conn-1234",
    "x-goog-content-length-range": "0,1073741824",
}


def _response(status: int = 200, body: Any = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json = MagicMock(return_value=body)
    response.text = "" if body is None else str(body)
    return response


def _upload_url_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "url": _SIGNED_URL,
        "method": "PUT",
        "headers": dict(_SIGNED_HEADERS),
        "expiresAt": "2024-01-02T05:15:00Z",
        "day": "2024-01-02",
    }
    body.update(overrides)
    return body


def _handler(
    *, url_response: MagicMock | None = None, put_response: MagicMock | None = None
) -> tuple[MagicMock, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a mock handler whose client records each upload-url POST and signed-URL PUT."""
    posts: list[dict[str, Any]] = []
    puts: list[dict[str, Any]] = []

    async def capture_post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        return url_response if url_response is not None else _response(body=_upload_url_body())

    async def capture_put(url, **kwargs):
        puts.append({"url": url, **kwargs})
        return put_response if put_response is not None else _response()

    handler = MagicMock()
    handler.close = AsyncMock()
    handler.client.post = capture_post
    handler.client.put = capture_put
    return handler, posts, puts


@pytest.mark.asyncio
async def test_should_skip_empty_content():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    with patch(MOCK_TARGET, side_effect=AssertionError("client initialized on empty content")):
        assert await dest.deliver(content=b"", time_window=_window(), filename="usage.csv") is None


@pytest.mark.asyncio
async def test_should_request_a_signed_url_for_the_window_day_then_put_the_csv():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    handler, posts, puts = _handler()

    with patch(MOCK_TARGET, return_value=handler):
        await dest.deliver(content=b"header\nrow1\n", time_window=_window(), filename="usage.csv")

    assert len(posts) == 1
    assert posts[0]["url"] == "https://ternary.test/external-cost-sources/v1/conn-1234/upload-url"
    assert posts[0]["headers"] == {"Authorization": "Bearer test-api-key"}
    assert posts[0]["json"] == {"day": "2024-01-02"}
    assert posts[0]["timeout"] == TERNARY_UPLOAD_TIMEOUT_SECONDS
    assert len(puts) == 1
    assert puts[0]["url"] == _SIGNED_URL
    assert puts[0]["content"] == b"header\nrow1\n"
    assert puts[0]["headers"] == _SIGNED_HEADERS
    assert "Authorization" not in puts[0]["headers"]
    handler.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_key_the_upload_on_the_window_start_day_not_its_end():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    one_day = FocusTimeWindow(start_time=start, end_time=start + timedelta(days=1), frequency="daily")
    handler, posts, _ = _handler()
    with patch(MOCK_TARGET, return_value=handler):
        await dest.deliver(content=b"h\nr\n", time_window=one_day, filename="usage.csv")
    assert posts[0]["json"] == {"day": "2024-01-02"}


@pytest.mark.asyncio
async def test_no_redirect_handler_disables_redirects_including_retry_client():
    handler = _NoRedirectHTTPHandler()
    retry_client = handler.create_client(timeout=1.0, event_hooks=None)
    try:
        assert handler.client.follow_redirects is False
        assert retry_client.follow_redirects is False
    finally:
        await retry_client.aclose()
        await handler.close()


@pytest.mark.asyncio
async def test_should_url_encode_the_connection_id():
    dest = FocusTernaryDestination(prefix="exports", config=_config(connection_id="conn+id~ok"))
    handler, posts, _ = _handler()
    with patch(MOCK_TARGET, return_value=handler):
        await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    assert posts[0]["url"].endswith("/external-cost-sources/v1/conn%2Bid~ok/upload-url")


@pytest.mark.asyncio
async def test_should_pass_tags_column_through_unstripped():
    """The Ternary sink must not drop/strip any columns (forwards Tags as-is)."""
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    handler, _, puts = _handler()

    content = b'ServiceName,Tags,x_unknown\nfoo,"{""team_id"": ""t1""}",keepme\n'
    with patch(MOCK_TARGET, return_value=handler):
        await dest.deliver(content=content, time_window=_window(), filename="usage.csv")

    assert puts[0]["content"] == content


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 404, 422, 429, 500])
async def test_should_raise_and_skip_the_put_when_the_upload_url_request_fails(status):
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    handler, _, puts = _handler(url_response=_response(status=status, body={"message": "nope"}))
    with patch(MOCK_TARGET, return_value=handler):
        with pytest.raises(RuntimeError, match=f"upload-url request for 2024-01-02 failed \\({status}\\)"):
            await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    assert puts == []
    handler.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 503])
async def test_should_raise_when_the_signed_put_fails(status):
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    handler, _, _ = _handler(put_response=_response(status=status, body="SignatureDoesNotMatch"))
    with patch(MOCK_TARGET, return_value=handler):
        with pytest.raises(RuntimeError, match=f"upload for 2024-01-02 failed \\({status}\\)"):
            await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    handler.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://storage.googleapis.com/b/days/c/2024-01-02.csv",
        "https://evil.example.com/b/days/c/2024-01-02.csv",
        "https://storage.googleapis.com.evil.example.com/b/o",
        "https://169.254.169.254/computeMetadata/v1/",
    ],
)
async def test_should_refuse_to_put_to_a_non_gcs_url(url):
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    handler, _, puts = _handler(url_response=_response(body=_upload_url_body(url=url)))
    with patch(MOCK_TARGET, return_value=handler):
        with pytest.raises(ValueError, match="HTTPS GCS URL"):
            await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    assert puts == []


@pytest.mark.asyncio
async def test_should_accept_a_virtual_hosted_gcs_url():
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    url = "https://ter-ecs-t-us.storage.googleapis.com/days/c/2024-01-02.csv?X-Goog-Signature=abc"
    handler, _, puts = _handler(url_response=_response(body=_upload_url_body(url=url)))
    with patch(MOCK_TARGET, return_value=handler):
        await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    assert puts[0]["url"] == url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"url": ""}, "missing 'url' or 'headers'"),
        ({"headers": None}, "missing 'url' or 'headers'"),
        ({"method": "POST"}, "unsupported upload method 'POST'"),
    ],
)
async def test_should_reject_an_incomplete_upload_url_response(overrides, match):
    dest = FocusTernaryDestination(prefix="exports", config=_config())
    body = _upload_url_body(**overrides)
    handler, _, puts = _handler(url_response=_response(body=body))
    with patch(MOCK_TARGET, return_value=handler):
        with pytest.raises(RuntimeError, match=match):
            await dest.deliver(content=b"h\nr\n", time_window=_window(), filename="usage.csv")
    assert puts == []


@pytest.mark.parametrize("url", ["http://api.ternary.app", "http://evil.example.com:8080", "ftp://ternary.test"])
def test_should_reject_non_https_base_url(url):
    with pytest.raises(ValueError, match="HTTPS"):
        FocusTernaryDestination(prefix="exports", config=_config(base_url=url))


@pytest.mark.parametrize("url", ["https://ternary.test", "http://localhost:8080", "http://127.0.0.1:8080"])
def test_should_accept_https_or_loopback_base_url(url):
    dest = FocusTernaryDestination(prefix="exports", config=_config(base_url=url))
    assert dest.base_url == url.rstrip("/")


def test_factory_creates_ternary_destination_from_config():
    from litellm.integrations.focus.destinations.factory import FocusDestinationFactory

    dest = FocusDestinationFactory.create(
        provider="ternary",
        prefix="exports",
        config={"api_key": "k", "connection_id": "c", "base_url": "https://ternary.test"},
    )
    assert isinstance(dest, FocusTernaryDestination)
    assert dest.connection_id == "c"


def test_factory_resolves_ternary_config_from_env(monkeypatch):
    monkeypatch.setenv("TERNARY_API_KEY", "envk")
    monkeypatch.setenv("TERNARY_CONNECTION_ID", "envc")
    monkeypatch.setenv("TERNARY_BASE_URL", "https://ternary.test")
    from litellm.integrations.focus.destinations.factory import FocusDestinationFactory

    dest = FocusDestinationFactory.create(provider="ternary", prefix="exports")
    assert isinstance(dest, FocusTernaryDestination)
    assert dest.api_key == "envk"
    assert dest.connection_id == "envc"


@pytest.mark.parametrize("missing", ["TERNARY_API_KEY", "TERNARY_CONNECTION_ID", "TERNARY_BASE_URL"])
def test_factory_requires_each_ternary_env_var(monkeypatch, missing):
    for var in ("TERNARY_API_KEY", "TERNARY_CONNECTION_ID", "TERNARY_BASE_URL"):
        monkeypatch.setenv(var, "https://ternary.test" if var == "TERNARY_BASE_URL" else "x")
    monkeypatch.delenv(missing, raising=False)
    from litellm.integrations.focus.destinations.factory import FocusDestinationFactory

    with pytest.raises(ValueError, match=missing):
        FocusDestinationFactory.create(provider="ternary", prefix="exports")
