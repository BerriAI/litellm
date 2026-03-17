"""
Tests that the timeout set via router_settings (which mirrors --request_timeout,
litellm_settings.request_timeout, or router_settings.timeout) actually reaches
the HTTP client in the Anthropic adapter-based messages path.

Uses a local aiohttp mock server that delays its response to verify that:
- A short timeout causes the request to fail with a timeout error.
- A sufficiently long timeout allows the request to succeed.
- Streaming requests honour the same timeout for the initial connection.

Each test loads its configuration from a temporary YAML file, mirroring
how the proxy initialises the Router from config.
"""

import asyncio
import os
import sys
import textwrap

import pytest
import yaml
from aiohttp import web

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm import Router


# ---------------------------------------------------------------------------
# Mock Anthropic /v1/messages endpoint with a fixed delay
# ---------------------------------------------------------------------------

MOCK_DELAY_SECONDS = 3


async def _delayed_messages_handler(request: web.Request) -> web.Response:
    """Return a minimal valid Anthropic messages response after a delay."""
    await asyncio.sleep(MOCK_DELAY_SECONDS)
    return web.json_response(
        {
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
    )


async def _delayed_streaming_messages_handler(
    request: web.Request,
) -> web.StreamResponse:
    """Return a streaming SSE Anthropic messages response after a delay.

    The delay happens *before* any bytes are sent, so the httpx timeout
    (which covers the period until response headers arrive) should trigger.
    """
    await asyncio.sleep(MOCK_DELAY_SECONDS)

    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
    )
    await response.prepare(request)

    events = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream_timeout_123","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello from streaming mock"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    for event in events:
        await response.write(event.encode())

    await response.write_eof()
    return response


@pytest.fixture()
async def mock_anthropic_server():
    """Start a local HTTP server that mimics a slow Anthropic API."""
    app = web.Application()
    app.router.add_post("/v1/messages", _delayed_messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


@pytest.fixture()
async def mock_anthropic_streaming_server():
    """Start a local HTTP server that mimics a slow *streaming* Anthropic API."""
    app = web.Application()
    app.router.add_post("/v1/messages", _delayed_streaming_messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_messages_timeout_too_short(mock_anthropic_server):
    """
    When the router timeout (1 s) is shorter than the server delay (3 s),
    the request must raise a timeout error.
    """
    config_yaml = textwrap.dedent(
        f"""\
        model_list:
          - model_name: test-model
            litellm_params:
              model: anthropic/claude-sonnet-4-5-20250929
              api_key: fake-key
              api_base: "{mock_anthropic_server}"
        router_settings:
          timeout: 1
          num_retries: 0
    """
    )
    router = _router_from_yaml(config_yaml)

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


@pytest.mark.asyncio
async def test_anthropic_messages_timeout_sufficient(mock_anthropic_server):
    """
    When the router timeout (10 s) is longer than the server delay (3 s),
    the request must succeed and return a valid response.
    """
    config_yaml = textwrap.dedent(
        f"""\
        model_list:
          - model_name: test-model
            litellm_params:
              model: anthropic/claude-sonnet-4-5-20250929
              api_key: fake-key
              api_base: "{mock_anthropic_server}"
        router_settings:
          timeout: 10
          num_retries: 0
    """
    )
    router = _router_from_yaml(config_yaml)

    response = await router.aanthropic_messages(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
    )

    assert response["id"] == "msg_test_timeout_123"
    assert response["content"][0]["text"] == "Hello from mock server"


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_streaming_timeout_too_short(mock_anthropic_streaming_server):
    """
    When the router timeout (1 s) is shorter than the server delay (3 s),
    a *streaming* request must raise a timeout error (the timeout fires
    before response headers arrive).
    """
    config_yaml = textwrap.dedent(
        f"""\
        model_list:
          - model_name: test-model
            litellm_params:
              model: anthropic/claude-sonnet-4-5-20250929
              api_key: fake-key
              api_base: "{mock_anthropic_streaming_server}"
        router_settings:
          timeout: 1
          num_retries: 0
    """
    )
    router = _router_from_yaml(config_yaml)

    with pytest.raises(Exception) as exc_info:
        resp = await router.aanthropic_messages(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            stream=True,
        )
        # If the call itself doesn't raise (some implementations return an
        # async iterator), consuming the first chunk should trigger it.
        if hasattr(resp, "__aiter__"):
            async for _ in resp:
                break

    exc = exc_info.value
    assert (
        "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
    ), f"Expected a timeout-related exception, got {type(exc).__name__}: {exc}"


@pytest.mark.asyncio
async def test_anthropic_streaming_timeout_sufficient(mock_anthropic_streaming_server):
    """
    When the router timeout (10 s) is longer than the server delay (3 s),
    a streaming request must succeed and yield SSE chunks.
    """
    config_yaml = textwrap.dedent(
        f"""\
        model_list:
          - model_name: test-model
            litellm_params:
              model: anthropic/claude-sonnet-4-5-20250929
              api_key: fake-key
              api_base: "{mock_anthropic_streaming_server}"
        router_settings:
          timeout: 10
          num_retries: 0
    """
    )
    router = _router_from_yaml(config_yaml)

    response = await router.aanthropic_messages(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        stream=True,
    )

    # The response is an async generator of SSE chunks; drain it.
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    # We should have received at least one chunk containing our test text.
    joined = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)
    assert b"Hello from streaming mock" in joined


@pytest.mark.asyncio
async def test_anthropic_streaming_stream_timeout_too_short(
    mock_anthropic_streaming_server,
):
    """
    When stream_timeout (1 s) is set explicitly and is shorter than the
    server delay (3 s), the streaming request must time out — even if the
    general timeout is generous.
    """
    config_yaml = textwrap.dedent(
        f"""\
        model_list:
          - model_name: test-model
            litellm_params:
              model: anthropic/claude-sonnet-4-5-20250929
              api_key: fake-key
              api_base: "{mock_anthropic_streaming_server}"
        router_settings:
          timeout: 30
          stream_timeout: 1
          num_retries: 0
    """
    )
    router = _router_from_yaml(config_yaml)

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
