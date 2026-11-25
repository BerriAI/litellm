import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines, iter_sse_lines
from litellm.llms.custom_httpx.stream_close import aclose_if_supported
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.responses.streaming_iterator import spawn_logging_task
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

if TYPE_CHECKING:
    from litellm.llms.base_llm.google_genai.transformation import (
        BaseGoogleGenAIGenerateContentConfig,
    )
else:
    BaseGoogleGenAIGenerateContentConfig = Any

GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ: Final = PassThroughEndpointLogging()


def _encode_google_genai_sse_event(event_lines: list[str]) -> bytes:
    return ("\n".join(event_lines) + "\n\n").encode("utf-8")


def _next_google_genai_sse_chunk(line_iter: Iterator[str]) -> bytes:
    event_lines: Final[list[str]] = []
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


async def _anext_google_genai_sse_chunk(line_iter: AsyncIterator[str]) -> bytes:
    event_lines: Final[list[str]] = []
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
        request_body: dict[str, object],
        model: str,
        custom_llm_provider: str,
        hidden_params: dict[str, object] | None = None,
    ):
        self.litellm_logging_obj = litellm_logging_obj
        self.request_body = request_body
        self.start_time = litellm_logging_obj.start_time
        self.collected_chunks: list[bytes] = []
        self.model = model
        self.custom_llm_provider = custom_llm_provider
        self.endpoint_type: Final = (
            EndpointType.GEMINI if custom_llm_provider == litellm.LlmProviders.GEMINI.value else EndpointType.VERTEX_AI
        )
        self._hidden_params: dict[str, object] = hidden_params or {}
        self._logging_dispatched: bool = False
        self._stream_failed: bool = False
        self._logging_task: asyncio.Task[object] | None = None

    def _should_log_collected_chunks(self) -> bool:
        if self._logging_dispatched or self._stream_failed:
            return False
        return bool(self.collected_chunks) and litellm.disable_streaming_logging is not True

    def _build_async_streaming_logging_coroutine(self) -> Coroutine[object, object, object]:
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        end_time: Final = datetime.now()
        return PassThroughStreamingHandler.route_streaming_logging_to_handler(
            litellm_logging_obj=self.litellm_logging_obj,
            passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
            url_route="/v1/generateContent",
            request_body=self.request_body or {},
            endpoint_type=self.endpoint_type,
            start_time=self.start_time,
            raw_bytes=tuple(self.collected_chunks),
            end_time=end_time,
            model=self.model,
        )

    def _schedule_async_streaming_logging(self) -> None:
        if self._logging_dispatched:
            return
        self._logging_dispatched = True
        self._logging_task = spawn_logging_task(
            asyncio.get_running_loop(),
            self._build_async_streaming_logging_coroutine(),
            task_name="Google GenAI stream success log",
        )

    async def _handle_async_streaming_logging(self) -> None:
        """Handle the logging after all chunks have been collected."""
        self._schedule_async_streaming_logging()


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
        litellm_metadata: Mapping[str, object],
        custom_llm_provider: str,
        request_body: dict[str, object] | None = None,
        hidden_params: dict[str, object] | None = None,
    ):
        super().__init__(
            litellm_logging_obj=logging_obj,
            request_body=request_body or {},
            model=model,
            custom_llm_provider=custom_llm_provider,
            hidden_params=hidden_params,
        )
        self.response = response
        self.model = model
        self.generate_content_provider_config = generate_content_provider_config
        self.litellm_metadata = litellm_metadata
        # Gemini streamGenerateContent uses SSE line framing; iter_sse_lines keeps
        # large inlineData payloads (e.g. image/jpeg) intact within one event.
        self.stream_iterator = iter_sse_lines(response.iter_text())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk: Final = _next_google_genai_sse_chunk(self.stream_iterator)
            self.collected_chunks.append(chunk)
            return chunk
        except StopIteration:
            raise StopIteration

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
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
        litellm_metadata: Mapping[str, object],
        custom_llm_provider: str,
        request_body: dict[str, object] | None = None,
        hidden_params: dict[str, object] | None = None,
    ):
        super().__init__(
            litellm_logging_obj=logging_obj,
            request_body=request_body or {},
            model=model,
            custom_llm_provider=custom_llm_provider,
            hidden_params=hidden_params,
        )
        self.response = response
        self.model = model
        self.generate_content_provider_config = generate_content_provider_config
        self.litellm_metadata = litellm_metadata
        # Gemini streamGenerateContent uses SSE line framing; aiter_sse_lines keeps
        # large inlineData payloads (e.g. image/jpeg) intact within one event.
        self.stream_iterator = aiter_sse_lines(response.aiter_text())

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            chunk: Final = await _anext_google_genai_sse_chunk(self.stream_iterator)
            if self.litellm_logging_obj.completion_start_time is None:
                self.litellm_logging_obj.update_completion_start_time(completion_start_time=datetime.now())
            self.collected_chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            await self._handle_async_streaming_logging()
            raise
        except asyncio.CancelledError:
            if self._should_log_collected_chunks():
                self._schedule_async_streaming_logging()
            raise
        except Exception:
            self._stream_failed = True
            raise

    async def log_collected_chunks_on_disconnect(self) -> bool:
        """Flush buffered success logging during shielded proxy disconnect cleanup."""
        if self._logging_dispatched:
            if self._logging_task is not None and not self._logging_task.done():
                try:
                    await self._logging_task
                except Exception as exc:  # noqa: BLE001  # disconnect cleanup must not fail stream teardown
                    verbose_logger.debug("Google GenAI stream success logging task failed on disconnect: %s", exc)
            return bool(self.litellm_logging_obj.model_call_details.get("has_dispatched_final_stream_success"))
        if not self._should_log_collected_chunks():
            return False
        self._logging_dispatched = True
        try:
            await self._build_async_streaming_logging_coroutine()
        except Exception as exc:  # noqa: BLE001  # disconnect cleanup must not fail stream teardown
            verbose_logger.debug("Google GenAI disconnect logging failed: %s", exc)
            return False
        return bool(self.litellm_logging_obj.model_call_details.get("has_dispatched_final_stream_success"))

    async def aclose(self) -> None:
        """Schedule pending success logging before closing the upstream SSE response."""
        try:
            if self._should_log_collected_chunks():
                self._schedule_async_streaming_logging()
        finally:
            try:
                await aclose_if_supported(self.stream_iterator)
            finally:
                await aclose_if_supported(self.response)


async def log_google_genai_stream_on_disconnect(response: object) -> bool | None:
    """Log collected chunks for a disconnected Google GenAI stream if `response` is one."""
    if not isinstance(response, AsyncGoogleGenAIGenerateContentStreamingIterator):
        return None
    return await response.log_collected_chunks_on_disconnect()
