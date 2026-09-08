import json
from collections.abc import AsyncIterator
from typing import Final, cast

import pytest

import litellm
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
MESSAGES: Final = [{"role": "user", "content": "Hello"}]
MESSAGES_RESPONSE: Final = {
    "id": "msg_native",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "Hello from native Messages"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 4},
}


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


async def call_messages(server: RecordingServer, **kwargs: object):
    return await litellm.anthropic.messages.acreate(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_key="test-key",
        api_base=server.base_url,
        **kwargs,
    )


def assert_native_request(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert "accept-encoding" not in server.requests[0].headers


@pytest.mark.asyncio
async def test_messages_sends_expected_provider_request(messages_server: RecordingServer) -> None:
    response: Final = await call_messages(messages_server)

    assert response["content"] == [{"type": "text", "text": "Hello from native Messages"}]
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-rust": "true"}
    assert_native_request(messages_server)
    request: Final = messages_server.requests[0]
    assert request.path == "/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.body == {
        "model": "claude-sonnet-4-5-20250929",
        "messages": MESSAGES,
        "max_tokens": 64,
    }


@pytest.mark.asyncio
async def test_messages_sends_custom_headers(messages_server: RecordingServer) -> None:
    await call_messages(messages_server, extra_headers={"x-trace-id": "trace-1"})

    assert messages_server.requests[0].headers["x-trace-id"] == "trace-1"


@pytest.mark.asyncio
async def test_messages_resolves_provider_credentials(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")

    await litellm.anthropic.messages.acreate(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_base=messages_server.base_url,
    )

    assert messages_server.requests[0].headers["x-api-key"] == "environment-key"


@pytest.mark.asyncio
async def test_messages_explicit_credentials_override_defaults(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")

    await call_messages(messages_server)

    assert messages_server.requests[0].headers["x-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_azure_messages_uses_foundry_endpoint_and_credentials(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(
        model="azure_ai/claude-opus-4.5",
        messages=MESSAGES,
        max_tokens=64,
        api_key="azure-key",
        api_base=f"{messages_server.base_url}/anthropic",
    )

    assert_native_request(messages_server)
    assert messages_server.requests[0].path == "/anthropic/v1/messages"
    assert messages_server.requests[0].headers["x-api-key"] == "azure-key"


@pytest.mark.asyncio
async def test_messages_stream_yields_anthropic_events(messages_server: RecordingServer) -> None:
    stream: Final = cast(AsyncIterator[bytes], await call_messages(messages_server, stream=True))
    payload: Final = b"".join([chunk async for chunk in stream])

    assert_native_request(messages_server)
    assert "stream" not in messages_server.requests[0].body
    assert b"event: message_start" in payload
    assert b"event: content_block_delta" in payload
    assert b"Hello from native Messages" in payload
    assert b"event: message_stop" in payload
    message_delta: Final = next(
        json.loads(block.split(b"data: ", 1)[1])
        for block in payload.split(b"\n\n")
        if block.startswith(b"event: message_delta")
    )
    assert message_delta["usage"] == {"input_tokens": 5, "output_tokens": 4}
