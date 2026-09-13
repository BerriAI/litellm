import re
from collections.abc import AsyncIterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    AnthropicMessagesStreamingResponse,
    BaseAnthropicMessagesStreamingIterator,
    _is_message_stop_chunk,
    _is_provider_error_chunk,
    aclose_if_supported,
)

if TYPE_CHECKING:
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

CACHED_STREAM_EVENTS_KEY: Final = "litellm_cached_anthropic_sse_events"

_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})

_SSE_EVENT_BOUNDARY: Final = re.compile(r"(?<=\n\n)")


def _decode(chunk: bytes | str) -> str:
    return chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk


def _split_sse_events(stream_text: str) -> tuple[str, ...]:
    return tuple(event for event in _SSE_EVENT_BOUNDARY.split(stream_text) if event)


class AnthropicMessagesStreamCacheWriter:
    def __init__(
        self,
        stream: AsyncIterator[bytes | str],
        caching_handler: "LLMCachingHandler",
    ) -> None:
        self.stream = stream
        self.caching_handler = caching_handler
        self.collected_chunks: list[bytes] = []  # mutable-ok: rebuilding a tuple per SSE chunk is quadratic
        self.persisted = False
        self._hidden_params: dict[str, object] = dict(  # mutable-ok: callers stamp cache_key in here
            stream._hidden_params if isinstance(stream, AnthropicMessagesStreamingResponse) else _EMPTY_MAPPING
        )

    @property
    def has_buffered_provider_output(self) -> bool:
        return getattr(self.stream, "has_buffered_provider_output", False) is True

    def __aiter__(self) -> "AnthropicMessagesStreamCacheWriter":
        return self

    async def __anext__(self) -> bytes | str:
        try:
            chunk: Final = await self.stream.__anext__()
        except StopAsyncIteration:
            await self._persist()
            raise
        self.collected_chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
        return chunk

    async def aclose(self) -> None:
        await aclose_if_supported(self.stream)

    async def _persist(self) -> None:
        if self.persisted or litellm.cache is None:
            return
        collected_stream: Final = b"".join(self.collected_chunks)
        if not _is_message_stop_chunk(collected_stream) or _is_provider_error_chunk(collected_stream):
            return
        self.persisted = True

        if not self.caching_handler._should_store_result_in_cache(
            original_function=self.caching_handler.original_function,
            kwargs=self.caching_handler.request_kwargs,
        ):
            return
        preset_cache_key: Final = self.caching_handler.preset_cache_key
        cache_key_override: Final[Mapping[str, object]] = (
            MappingProxyType({"cache_key": preset_cache_key}) if preset_cache_key is not None else _EMPTY_MAPPING
        )
        request_kwargs: Final[Mapping[str, object]] = MappingProxyType(
            {**self.caching_handler.request_kwargs, **cache_key_override}
        )

        try:
            events: Final = _split_sse_events(collected_stream.decode("utf-8"))
            cached_payload: Final = {
                CACHED_STREAM_EVENTS_KEY: events
            }  # mutable-ok: cache backends serialize plain dicts
            await litellm.cache.async_add_cache(
                cached_payload,
                dynamic_cache_object=self.caching_handler.dual_cache,
                **request_kwargs,
            )
        except Exception as e:  # noqa: BLE001  # a cache write must never surface as a client-visible stream error
            verbose_logger.exception("Anthropic Messages stream cache write failed: %s", e)


class CachedAnthropicMessagesStreamIterator(BaseAnthropicMessagesStreamingIterator):
    def __init__(
        self,
        events: Sequence[str],
        litellm_logging_obj: "LiteLLMLoggingObj",
        request_body: Mapping[str, object],
    ) -> None:
        body: Final = dict(request_body)  # mutable-ok: the base iterator takes a plain dict
        super().__init__(litellm_logging_obj=litellm_logging_obj, request_body=body)
        self.chunks: Final[tuple[bytes, ...]] = tuple(event.encode("utf-8") for event in events)
        self.current_index = 0
        self.logged = False
        self._hidden_params: dict[str, object] = {"cache_hit": True}  # mutable-ok: callers stamp cache_key in here
        litellm_logging_obj.model_call_details["cache_hit"] = True

    def __aiter__(self) -> "CachedAnthropicMessagesStreamIterator":
        return self

    async def __anext__(self) -> bytes:
        if self.current_index >= len(self.chunks):
            if not self.logged:
                self.logged = True
                chunks: Final = list(self.chunks)  # mutable-ok: the logging handler takes a list
                await self._handle_streaming_logging(chunks)
            raise StopAsyncIteration
        chunk: Final = self.chunks[self.current_index]
        self.current_index += 1
        return chunk


def get_cached_stream_events(cached_result: Mapping[str, object]) -> tuple[str, ...] | None:
    events: Final = cached_result.get(CACHED_STREAM_EVENTS_KEY)
    if isinstance(events, (list, tuple)):
        return tuple(_decode(event) for event in events if isinstance(event, (bytes, str)))
    return None


def convert_cached_anthropic_messages_result(
    cached_result: Mapping[str, object],
    logging_obj: "LiteLLMLoggingObj",
    kwargs: Mapping[str, object],
) -> Mapping[str, object] | CachedAnthropicMessagesStreamIterator:
    events: Final = get_cached_stream_events(cached_result)
    if events is None:
        return cached_result
    return CachedAnthropicMessagesStreamIterator(
        events=events,
        litellm_logging_obj=logging_obj,
        request_body=kwargs,
    )
