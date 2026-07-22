import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

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

# Buffer raw bytes and split only on real SSE frame delimiters. httpx
# ``aiter_lines`` splits on ``str.splitlines`` boundaries (U+2028, U+2029,
# U+0085, form feed, ...) which Gemini emits raw inside ``data:`` JSON, so it
# would slice an event mid-payload and corrupt the JSON the SDK then parses.
_SSE_FRAME_DELIMITERS: Tuple[bytes, ...] = (b"\r\n\r\n", b"\n\n", b"\r\r")


def _split_sse_frame(buffer: bytes) -> Tuple[Optional[bytes], bytes]:
    """Pop the first complete SSE frame (delimiter included) from ``buffer``."""
    frame_starts = (
        (position, delimiter)
        for delimiter in _SSE_FRAME_DELIMITERS
        if (position := buffer.find(delimiter)) != -1
    )
    first = min(frame_starts, key=lambda item: item[0], default=None)
    if first is None:
        return None, buffer
    position, delimiter = first
    frame_end = position + len(delimiter)
    return buffer[:frame_end], buffer[frame_end:]


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
        response: httpx.Response,
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
        self.stream_iterator = response.iter_bytes()
        self._buffer: bytes = b""

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        while True:
            frame, self._buffer = _split_sse_frame(self._buffer)
            if frame is not None:
                self.collected_chunks.append(frame)
                return frame
            try:
                self._buffer += next(self.stream_iterator)
            except StopIteration:
                if self._buffer:
                    frame, self._buffer = self._buffer, b""
                    self.collected_chunks.append(frame)
                    return frame
                raise

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
        response: httpx.Response,
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
        self.stream_iterator = response.aiter_bytes()
        self._buffer: bytes = b""

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        while True:
            frame, self._buffer = _split_sse_frame(self._buffer)
            if frame is not None:
                self.collected_chunks.append(frame)
                return frame
            try:
                self._buffer += await self.stream_iterator.__anext__()
            except StopAsyncIteration:
                if self._buffer:
                    frame, self._buffer = self._buffer, b""
                    self.collected_chunks.append(frame)
                    return frame
                await self._handle_async_streaming_logging()
                raise
