"""The Responses API stream a router hands back when its attempts' streams can fall back mid-stream."""

import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Final

from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    _get_openai_response_types,  # pyright: ignore[reportPrivateUsage]  # the lazily imported OpenAI event types
)
from litellm.router_utils.add_retry_fallback_headers import prepare_fallback_hidden_params

_OPENAI_TYPES: Final = _get_openai_response_types()
_RESPONSES_TERMINAL_EVENT_TYPES: Final = (
    _OPENAI_TYPES.ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    _OPENAI_TYPES.ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
    _OPENAI_TYPES.ResponsesAPIStreamEvents.RESPONSE_FAILED,
)


class FallbackResponsesStreamWrapper(BaseResponsesAPIStreamingIterator):
    """
    Subclasses BaseResponsesAPIStreamingIterator only for isinstance compatibility (proxy and
    interactions code paths check the type). Bypasses the parent constructor and delegates
    iteration to an async generator.
    """

    fallback_headers_adopted: bool = False

    def __init__(self, async_generator: AsyncGenerator[object, None], source_iterator: object) -> None:
        self._async_generator: Final = async_generator
        self._source_iterator: Final = source_iterator
        # The bridge path (LiteLLMCompletionStreamingIterator) skips super().__init__ and lacks
        # many of these attributes, and stores its logging object as `litellm_logging_obj`.
        self.response = getattr(source_iterator, "response", None)
        self.model = getattr(source_iterator, "model", None)
        self.logging_obj = getattr(
            source_iterator,
            "logging_obj",
            getattr(source_iterator, "litellm_logging_obj", None),
        )
        self.finished = False
        self.responses_api_provider_config = getattr(source_iterator, "responses_api_provider_config", None)
        self.completed_response = None
        self.start_time = getattr(source_iterator, "start_time", datetime.now())
        self._failure_handled = False
        self._yielded_first_chunk = False
        self._generated_content = ""
        self._completed_response_cached = False
        self._completed_response_logged = False
        self._completed_response_cache_hit = None
        self._persist_completed_response_before_logging = True
        self._stream_created_time = time.time()
        self.litellm_metadata = getattr(source_iterator, "litellm_metadata", None)
        self.custom_llm_provider = getattr(source_iterator, "custom_llm_provider", None)
        self.request_data = getattr(source_iterator, "request_data", {}) or {}
        self.call_type = getattr(source_iterator, "call_type", None)
        self._hidden_params = dict(getattr(source_iterator, "_hidden_params", None) or {})

    def adopt_fallback_headers(self, fallback_response: object) -> tuple[dict[str, object], dict[str, object]]:
        prepared: Final = prepare_fallback_hidden_params(fallback_response)
        self._hidden_params = {**prepared[0], "additional_headers": prepared[1]}
        self.fallback_headers_adopted = True
        return prepared

    def __aiter__(self) -> "FallbackResponsesStreamWrapper":
        return self

    async def __anext__(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # yields whatever the wrapped stream yields
        try:
            chunk: Final = await self._async_generator.__anext__()
        except StopAsyncIteration:
            # The bridge path emits its final response.completed and then stops, so a terminal
            # event this wrapper never saw is read off whatever the source iterator latched.
            if self.completed_response is None:
                self.completed_response = getattr(self._source_iterator, "completed_response", None)
            raise
        # The proxy's container-ownership hook reads `completed_response` off the stream it got (#30210)
        if self.completed_response is None and getattr(chunk, "type", None) in _RESPONSES_TERMINAL_EVENT_TYPES:
            self.completed_response = chunk  # pyright: ignore[reportAttributeAccessIssue]  # the terminal event itself
        return chunk

    async def aclose(self) -> None:
        await self._async_generator.aclose()
