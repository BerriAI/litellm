from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    BaseAnthropicMessagesStreamingIterator,
    _is_message_stop_chunk,
    _is_provider_error_chunk,
    aclose_if_supported,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

if TYPE_CHECKING:
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LLMCachingHandler = Any
    LiteLLMLoggingObj = Any

CACHED_STREAM_EVENTS_KEY = "litellm_cached_anthropic_sse_events"


def _decode(chunk: bytes | str) -> str:
    return chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk


class AnthropicMessagesStreamCacheWriter:
    def __init__(
        self,
        stream: AsyncIterator[bytes | str],
        caching_handler: "LLMCachingHandler",
    ) -> None:
        self.stream = stream
        self.caching_handler = caching_handler
        self.collected_events: list[str] = []
        self.saw_message_stop = False
        self.saw_provider_error = False
        self.persisted = False
        self._hidden_params: dict[str, Any] = getattr(stream, "_hidden_params", {}) or {}

    def __aiter__(self) -> "AnthropicMessagesStreamCacheWriter":
        return self

    async def __anext__(self) -> bytes | str:
        try:
            chunk = await self.stream.__anext__()
        except StopAsyncIteration:
            await self._persist()
            raise
        chunk_bytes = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        self.saw_message_stop = self.saw_message_stop or _is_message_stop_chunk(chunk_bytes)
        self.saw_provider_error = self.saw_provider_error or _is_provider_error_chunk(chunk_bytes)
        self.collected_events.append(_decode(chunk))
        return chunk

    async def aclose(self) -> None:
        await aclose_if_supported(self.stream)

    async def _persist(self) -> None:
        if self.persisted or litellm.cache is None:
            return
        if not self.saw_message_stop or self.saw_provider_error:
            return
        self.persisted = True

        request_kwargs = dict(self.caching_handler.request_kwargs)
        if not self.caching_handler._should_store_result_in_cache(
            original_function=self.caching_handler.original_function,
            kwargs=request_kwargs,
        ):
            return
        preset_cache_key = self.caching_handler.preset_cache_key
        if preset_cache_key is not None:
            request_kwargs["cache_key"] = preset_cache_key

        try:
            await litellm.cache.async_add_cache(
                {CACHED_STREAM_EVENTS_KEY: self.collected_events},
                dynamic_cache_object=self.caching_handler.dual_cache,
                **request_kwargs,
            )
        except Exception as e:
            verbose_logger.exception("Anthropic Messages stream cache write failed: %s", e)


class CachedAnthropicMessagesStreamIterator(BaseAnthropicMessagesStreamingIterator):
    def __init__(
        self,
        events: list[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict[str, Any],
    ) -> None:
        super().__init__(litellm_logging_obj=litellm_logging_obj, request_body=request_body)
        self.chunks: list[bytes] = [event.encode("utf-8") for event in events]
        self.current_index = 0
        self.logged = False
        self._hidden_params: dict[str, Any] = {"cache_hit": True}
        litellm_logging_obj.model_call_details["cache_hit"] = True

    def __aiter__(self) -> "CachedAnthropicMessagesStreamIterator":
        return self

    async def __anext__(self) -> bytes:
        if self.current_index >= len(self.chunks):
            if not self.logged:
                self.logged = True
                await self._handle_streaming_logging(self.chunks)
            raise StopAsyncIteration
        chunk = self.chunks[self.current_index]
        self.current_index += 1
        return chunk


def get_cached_stream_events(cached_result: dict[str, Any]) -> list[str] | None:
    events = cached_result.get(CACHED_STREAM_EVENTS_KEY)
    if isinstance(events, list):
        return [_decode(event) for event in events]
    return None


def convert_cached_anthropic_messages_result(
    cached_result: dict[str, Any],
    logging_obj: LiteLLMLoggingObj,
    kwargs: dict[str, Any],
) -> AnthropicMessagesResponse | CachedAnthropicMessagesStreamIterator:
    events = get_cached_stream_events(cached_result)
    if events is not None:
        return CachedAnthropicMessagesStreamIterator(
            events=events,
            litellm_logging_obj=logging_obj,
            request_body=kwargs,
        )
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict; validating would drop provider fields we must replay verbatim
        AnthropicMessagesResponse, cached_result
    )
