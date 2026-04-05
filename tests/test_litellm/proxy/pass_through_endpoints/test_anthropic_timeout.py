"""
Tests that the timeout set via router_settings (which mirrors --request_timeout,
litellm_settings.request_timeout, or router_settings.timeout) actually reaches
the HTTP client in the Anthropic adapter-based messages path.

Uses mocked httpx calls to verify that:
- A short timeout causes the request to fail with a timeout error.
- A sufficiently long timeout allows the request to succeed.
- Streaming requests honour the same timeout for the initial connection.
- stream_timeout takes precedence over timeout for streaming requests.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm import Router

MOCK_TARGET = "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"


# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------


def _mock_anthropic_response() -> MagicMock:
    """Return a mock httpx.Response for a non-streaming Anthropic messages call."""
    response_body = {
        "id": "msg_test_timeout_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "Hello from mock server"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
        },
    }
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = response_body
    mock_resp.text = str(response_body)
    mock_resp.headers = {"content-type": "application/json"}
    return mock_resp


def _mock_streaming_response() -> MagicMock:
    """Return a mock httpx.Response for a streaming Anthropic messages call."""
    sse_events = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream_timeout_123","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello from streaming mock"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def _aiter_bytes():
        for event in sse_events:
            yield event.encode()

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/event-stream"}
    mock_resp.aiter_bytes = _aiter_bytes
    return mock_resp


# ---------------------------------------------------------------------------
# Helper – build a Router from a YAML config string
# ---------------------------------------------------------------------------


def _router_from_yaml(yaml_text: str) -> Router:
    """
    Parse a YAML config and construct a Router the same way the proxy does
    (see proxy_server.py around line 3140-3225).
    """
    config = yaml.safe_load(yaml_text)
    model_list = config.get("model_list", [])
    router_settings = config.get("router_settings", {})
    return Router(model_list=model_list, **router_settings)


def _make_config(**router_settings) -> str:
    """Build a YAML config string with the given router_settings."""
    config = {
        "model_list": [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "fake-key",
                    "api_base": "http://mock-api.local",
                },
            }
        ],
        "router_settings": router_settings,
    }
    return yaml.dump(config)


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_messages_timeout_too_short():
    """
    When the router timeout (1 s) is configured and the HTTP call raises a
    timeout exception, the error must propagate to the caller.
    """
    config_yaml = _make_config(timeout=1, num_retries=0)
    router = _router_from_yaml(config_yaml)

    with patch(MOCK_TARGET, new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timeout")

        with pytest.raises(Exception) as exc_info:
            await router.aanthropic_messages(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )

        exc = exc_info.value
        assert (
            "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
        ), f"Expected a timeout-related exception, got {type(exc).__name__}: {exc}"

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 1


@pytest.mark.asyncio
async def test_anthropic_messages_timeout_sufficient():
    """
    When the router timeout (10 s) is configured and the HTTP call succeeds,
    the response must be returned correctly with the right timeout passed.
    """
    config_yaml = _make_config(timeout=10, num_retries=0)
    router = _router_from_yaml(config_yaml)

    with patch(MOCK_TARGET, new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _mock_anthropic_response()

        response = await router.aanthropic_messages(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )

    assert response["id"] == "msg_test_timeout_123"
    assert response["content"][0]["text"] == "Hello from mock server"

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 10


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_streaming_timeout_too_short():
    """
    When the router timeout (1 s) is configured and the streaming HTTP call
    raises a timeout exception, the error must propagate to the caller.
    """
    config_yaml = _make_config(timeout=1, num_retries=0)
    router = _router_from_yaml(config_yaml)

    with patch(MOCK_TARGET, new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timeout")

        with pytest.raises(Exception) as exc_info:
            resp = await router.aanthropic_messages(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                stream=True,
            )
            if hasattr(resp, "__aiter__"):
                async for _ in resp:
                    break

        exc = exc_info.value
        assert (
            "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
        ), f"Expected a timeout-related exception, got {type(exc).__name__}: {exc}"

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 1


@pytest.mark.asyncio
async def test_anthropic_streaming_timeout_sufficient():
    """
    When the router timeout (10 s) is configured and the streaming HTTP call
    succeeds, the response chunks must contain the expected text.
    """
    config_yaml = _make_config(timeout=10, num_retries=0)
    router = _router_from_yaml(config_yaml)

    with patch(MOCK_TARGET, new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _mock_streaming_response()

        response = await router.aanthropic_messages(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            stream=True,
        )

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

    joined = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)
    assert b"Hello from streaming mock" in joined

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 10


@pytest.mark.asyncio
async def test_anthropic_streaming_stream_timeout_too_short():
    """
    When stream_timeout (1 s) is set explicitly and is shorter than the
    general timeout (30 s), stream_timeout must take precedence for streaming
    requests.
    """
    config_yaml = _make_config(timeout=30, stream_timeout=1, num_retries=0)
    router = _router_from_yaml(config_yaml)

    with patch(MOCK_TARGET, new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timeout")

        with pytest.raises(Exception) as exc_info:
            resp = await router.aanthropic_messages(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                stream=True,
            )
            if hasattr(resp, "__aiter__"):
                async for _ in resp:
                    break

        exc = exc_info.value
        assert (
            "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
        ), f"Expected a timeout-related exception, got {type(exc).__name__}: {exc}"

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 1
