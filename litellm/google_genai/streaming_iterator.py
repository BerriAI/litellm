import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

if TYPE_CHECKING:
    from litellm.llms.base_llm.google_genai.transformation import (
        BaseGoogleGenAIGenerateContentConfig,
    )
else:
    BaseGoogleGenAIGenerateContentConfig = Any

GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ = PassThroughEndpointLogging()


def _encode_google_genai_sse_event(event_lines: List[str]) -> bytes:
    return ("\n".join(event_lines) + "\n\n").encode("utf-8")


def _next_google_genai_sse_chunk(line_iter) -> bytes:
    event_lines: List[str] = []
    while True:
        try:
            line = next(line_iter)
        except StopIteration:
            if event_lines:
                return _encode_google_genai_sse_event(event_lines)
            raise
        if line == "":
            if event_lines:
                return _encode_google_genai_sse_event(event_lines)
            continue
        event_lines.append(line)


async def _anext_google_genai_sse_chunk(line_iter) -> bytes:
    event_lines: List[str] = []
    while True:
        try:
            line = await line_iter.__anext__()
        except StopAsyncIteration:
            if event_lines:
                return _encode_google_genai_sse_event(event_lines)
            raise
        if line == "":
            if event_lines:
                return _encode_google_genai_sse_event(event_lines)
            continue
        event_lines.append(line)


class BaseGoogleGenAIGenerateContentStreamingIterator:
    """
    Base class for Google GenAI Generate Content streaming iterators that provides common logic
    for streaming response handling and logging.
    """

    def __init__(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
        model: str,
        hidden_params: Optional[Dict[str, Any]] = None,
    ):
        self.litellm_logging_obj = litellm_logging_obj
        self.request_body = request_body
        self.start_time = datetime.now()
        self.collected_chunks: List[bytes] = []
        self.model = model
        self._hidden_params: Dict[str, Any] = hidden_params or {}

    async def _handle_async_streaming_logging(
        self,
    ):
        """Handle the logging after all chunks have been collected."""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        end_time = datetime.now()
        asyncio.create_task(
            PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=self.litellm_logging_obj,
                passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
                url_route="/v1/generateContent",
                request_body=self.request_body or {},
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=self.start_time,
                raw_bytes=self.collected_chunks,
                end_time=end_time,
                model=self.model,
            )
        )


class GoogleGenAIGenerateContentStreamingIterator(BaseGoogleGenAIGenerateContentStreamingIterator):
    """
    Streaming iterator specifically for Google GenAI generate content API.
    """

    def __init__(
        self,
        response,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        generate_content_provider_config: BaseGoogleGenAIGenerateContentConfig,
        litellm_metadata: dict,
        custom_llm_provider: str,
        request_body: Optional[dict] = None,
        hidden_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            litellm_logging_obj=logging_obj,
            request_body=request_body or {},
            model=model,
            hidden_params=hidden_params,
        )
        self.response = response
        self.model = model
        self.generate_content_provider_config = generate_content_provider_config
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider
        # Gemini streamGenerateContent uses SSE line framing; iter_lines keeps
        # large inlineData payloads (e.g. image/jpeg) intact within one event.
        self.stream_iterator = response.iter_lines()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = _next_google_genai_sse_chunk(self.stream_iterator)
            self.collected_chunks.append(chunk)
            return chunk
        except StopIteration:
            raise StopIteration

    def __aiter__(self):
        return self

    async def __anext__(self):
        # This should not be used for sync responses
        # If you need async iteration, use AsyncGoogleGenAIGenerateContentStreamingIterator
        raise NotImplementedError("Use AsyncGoogleGenAIGenerateContentStreamingIterator for async iteration")


class AsyncGoogleGenAIGenerateContentStreamingIterator(BaseGoogleGenAIGenerateContentStreamingIterator):
    """
    Async streaming iterator specifically for Google GenAI generate content API.
    """

    def __init__(
        self,
        response,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        generate_content_provider_config: BaseGoogleGenAIGenerateContentConfig,
        litellm_metadata: dict,
        custom_llm_provider: str,
        request_body: Optional[dict] = None,
        hidden_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            litellm_logging_obj=logging_obj,
            request_body=request_body or {},
            model=model,
            hidden_params=hidden_params,
        )
        self.response = response
        self.model = model
        self.generate_content_provider_config = generate_content_provider_config
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider
        # Gemini streamGenerateContent uses SSE line framing; aiter_lines keeps
        # large inlineData payloads (e.g. image/jpeg) intact within one event.
        self.stream_iterator = response.aiter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await _anext_google_genai_sse_chunk(self.stream_iterator)
            self.collected_chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            await self._handle_async_streaming_logging()
            raise StopAsyncIteration
