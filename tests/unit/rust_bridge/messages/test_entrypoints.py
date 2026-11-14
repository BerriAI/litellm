from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.messages.native.agentic_streaming_iterator import AgenticAnthropicStreamingIterator
from litellm.llms.anthropic.messages.native.response_cache import (
    AnthropicMessagesStreamCacheWriter,
    CachedAnthropicMessagesStreamIterator,
)
from litellm.llms.anthropic.messages.native.streaming_iterator import AnthropicMessagesStreamingResponse
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_EVENTS
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

_STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)
_SENT: Final = b"".join(_STREAM.payloads())


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    with recording_service() as recording:
        yield recording


def _azure_ai_call(server: RecordingServer, stream: object = True, call_id: str = "call") -> NativeCall:
    supplied: Final[dict[str, object]] = {
        "litellm_call_id": call_id,
        "model": "azure_ai/claude-sonnet-4-5",
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 8,
        "stream": stream,
        "api_key": "test-key",
        "api_base": server.base_url,
    }
    return NativeCall(args=(), kwargs=supplied, base={"custom_llm_provider": None, **supplied})


async def _amessages(call: NativeCall) -> object:
    route: Final = NATIVE_AMESSAGES.load()
    assert route is not None
    return await route(call)


async def _read(stream: object) -> bytes:
    assert isinstance(stream, AnthropicMessagesStreamingResponse | CachedAnthropicMessagesStreamIterator)
    return b"".join([bytes(chunk) async for chunk in stream])


@pytest.mark.asyncio
async def test_an_azure_ai_stream_is_the_streaming_response_python_returns(server: RecordingServer) -> None:
    server.enqueue(_STREAM)

    stream: Final = await _amessages(_azure_ai_call(server))

    assert type(stream) is AnthropicMessagesStreamingResponse
    assert stream._hidden_params["litellm_model_name"] == "azure_ai/claude-sonnet-4-5"  # pyright: ignore[reportPrivateUsage]  # the public stream's own hidden params
    assert await _read(stream) == _SENT


@pytest.mark.asyncio
async def test_an_azure_ai_stream_is_cached_on_a_miss_and_replayed_on_a_hit_without_the_provider(
    monkeypatch: pytest.MonkeyPatch, server: RecordingServer
) -> None:
    monkeypatch.setattr(
        litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=["anthropic_messages"])
    )
    server.expected_requests = 1
    server.enqueue(_STREAM)

    missed: Final = await _amessages(_azure_ai_call(server))

    assert isinstance(missed, AnthropicMessagesStreamCacheWriter)
    assert missed.persisted is True
    assert b"".join([bytes(chunk) async for chunk in missed]) == _SENT
    await asyncio.sleep(0)
    await asyncio.gather(*_PENDING_CACHE_WRITES)

    hit: Final = await _amessages(_azure_ai_call(server, call_id="hit"))

    assert isinstance(hit, CachedAnthropicMessagesStreamIterator)
    assert hit._hidden_params["cache_hit"] is True  # pyright: ignore[reportPrivateUsage]  # the replay's own hidden params
    assert isinstance(hit._hidden_params["cache_key"], str)  # pyright: ignore[reportPrivateUsage]  # the replay's own hidden params
    assert hit._hidden_params["litellm_call_id"] == "hit"  # pyright: ignore[reportPrivateUsage]  # the replay's own hidden params
    assert await _read(hit) == _SENT


class _AgenticHook(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.streams: Final[list[object]] = []

    async def async_should_run_agentic_loop(  # pyright: ignore[reportIncompatibleMethodOverride]  # the hook only records that it ran
        self,
        response: object,
        *args: object,
        **kwargs: object,  # kwargs-ok: the handler's hook arguments
    ) -> tuple[bool, dict[str, object]]:
        self.streams.append(response)
        return False, {}


@pytest.mark.asyncio
async def test_an_agentic_hook_reads_the_stream_through_python_s_agentic_iterator(
    monkeypatch: pytest.MonkeyPatch, server: RecordingServer
) -> None:
    hook: Final = _AgenticHook()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    server.enqueue(_STREAM)

    stream: Final = await _amessages(_azure_ai_call(server))

    assert isinstance(stream, AnthropicMessagesStreamingResponse)
    assert isinstance(stream.completion_stream, AgenticAnthropicStreamingIterator)
    assert await _read(stream) == _SENT
    assert len(hook.streams) == 1


class _ReplacingDeploymentHook(CustomLogger):
    replacement: Final = {"id": "replaced"}

    async def async_post_call_success_deployment_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # returns the replacement only
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> object:
        return self.replacement


@pytest.mark.asyncio
async def test_a_stream_the_caller_did_not_ask_for_is_delivered_through_the_deployment_success_hook(
    monkeypatch: pytest.MonkeyPatch, server: RecordingServer
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_ReplacingDeploymentHook()])
    server.enqueue(_STREAM)

    assert await _amessages(_azure_ai_call(server, stream=1)) is _ReplacingDeploymentHook.replacement
