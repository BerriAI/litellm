import asyncio
import datetime
from collections.abc import AsyncIterator, Callable
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES, LLMCachingHandler
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.messages.native.response_cache import AnthropicMessagesStreamCacheWriter
from litellm.rust_bridge.host.cache import (
    caching_handler,
    configured_cache,
    store_response,
    store_verbatim,
    wrap_stream,
)
from litellm.types.caching import LiteLLMCacheType


def test_configured_cache_preserves_identity_and_observes_runtime_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    initial: Final = Cache(type=LiteLLMCacheType.LOCAL)
    replacement: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", initial)
    assert configured_cache() is initial

    monkeypatch.setattr(litellm, "cache", replacement)
    assert configured_cache() is replacement

    monkeypatch.setattr(litellm, "cache", None)
    assert configured_cache() is None


_REQUEST: Final = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}


async def _stored(response: dict[str, object], store: Callable[..., None] = store_response) -> object:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    store(cache, response, **_REQUEST)
    await asyncio.gather(*_PENDING_CACHE_WRITES)
    return await cache.async_get_cache(**_REQUEST)


@pytest.mark.asyncio
async def test_store_response_drops_only_the_rust_marker_header_from_the_cached_entry() -> None:
    stored: Final = await _stored(
        {
            "id": "msg_1",
            "_hidden_params": {"additional_headers": {"x-litellm-rust": "true", "llm_provider-x": "1"}, "model": "m"},
        }
    )
    assert stored == {
        "id": "msg_1",
        "_hidden_params": {"additional_headers": {"llm_provider-x": "1"}, "model": "m"},
    }


@pytest.mark.asyncio
async def test_store_response_keeps_an_entry_without_the_rust_marker_unchanged() -> None:
    response: Final = {"id": "msg_2", "_hidden_params": {"additional_headers": {"llm_provider-x": "1"}}}
    assert await _stored(response) == response


async def _events() -> AsyncIterator[bytes]:
    yield b"event: message_stop\ndata: {}\n\n"


async def anthropic_messages() -> None:
    return None


def _logging() -> Logging:
    return Logging(
        model="claude-sonnet-4-5",
        messages=[],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="messages",
        function_id="messages",
    )


def test_wrap_stream_hands_the_stream_to_the_cache_writer_the_client_wrapper_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=["anthropic_messages"])
    monkeypatch.setattr(litellm, "cache", cache)
    stream: Final = _events()

    wrapped: Final = wrap_stream(cache, anthropic_messages, stream, None, **_REQUEST, litellm_logging_obj=_logging())

    assert isinstance(wrapped, AnthropicMessagesStreamCacheWriter)
    assert wrapped.stream is stream


def test_wrap_stream_leaves_a_stream_the_cache_would_not_store(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=["anthropic_messages"])
    monkeypatch.setattr(litellm, "cache", cache)
    stream: Final = _events()

    assert (
        wrap_stream(
            cache,
            anthropic_messages,
            stream,
            None,
            **_REQUEST,
            cache={"no-store": True},
            litellm_logging_obj=_logging(),
        )
        is stream
    )


@pytest.mark.asyncio
async def test_store_verbatim_keeps_the_rust_marker_header_in_the_cached_entry() -> None:
    response: Final[dict[str, object]] = {
        "id": "msg_3",
        "_hidden_params": {"additional_headers": {"x-litellm-rust": "true", "llm_provider-x": "1"}},
    }
    assert await _stored(response, store_verbatim) == response


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [store_response, store_verbatim])
@pytest.mark.parametrize("content", [[], [{"type": "text", "text": ""}]])
async def test_a_message_without_output_is_never_stored(
    store: Callable[..., None], content: list[dict[str, object]]
) -> None:
    assert await _stored({"id": "msg_4", "type": "message", "content": content}, store) is None


def test_wrap_stream_writes_through_the_handler_that_read_the_cache_over_a_later_call_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=["anthropic_messages"])
    monkeypatch.setattr(litellm, "cache", cache)
    logging_obj: Final = _logging()
    read: Final = LLMCachingHandler(
        original_function=anthropic_messages, request_kwargs=dict(_REQUEST), start_time=datetime.datetime.now()
    )
    logging_obj.llm_caching_handler = LLMCachingHandler(
        original_function=anthropic_messages, request_kwargs={}, start_time=datetime.datetime.now()
    )

    wrapped: Final = wrap_stream(
        cache, anthropic_messages, _events(), read, **_REQUEST, litellm_logging_obj=logging_obj
    )

    assert isinstance(wrapped, AnthropicMessagesStreamCacheWriter)
    assert wrapped.caching_handler is read


def test_caching_handler_is_the_one_the_logging_object_holds() -> None:
    logging_obj: Final = _logging()
    assert caching_handler(litellm_logging_obj=logging_obj) is None
    handler: Final = LLMCachingHandler(
        original_function=anthropic_messages, request_kwargs={}, start_time=datetime.datetime.now()
    )
    logging_obj.llm_caching_handler = handler
    assert caching_handler(litellm_logging_obj=logging_obj) is handler
