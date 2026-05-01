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

    def _build_logging_kwargs(self) -> dict:
        """
        Build the kwargs passed into the streaming logging route.

        Tags chunks as ``EndpointType.GOOGLE_GENAI`` (not ``VERTEX_AI``) so
        ``_route_streaming_logging_to_handler`` dispatches to
        ``GeminiPassthroughLoggingHandler`` instead of falling through with no
        match (which silently skipped every success_callback before this fix).

        The ``url_route`` includes the real ``/models/{model}:streamGenerateContent``
        segment so downstream URL parsing in ``GeminiPassthroughLoggingHandler``
        recovers the model name when the explicit ``model`` kwarg is absent.
        """
        return dict(
            litellm_logging_obj=self.litellm_logging_obj,
            passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
            url_route=f"/models/{self.model}:streamGenerateContent",
            request_body=self.request_body or {},
            endpoint_type=EndpointType.GOOGLE_GENAI,
            start_time=self.start_time,
            raw_bytes=self.collected_chunks,
            end_time=datetime.now(),
            model=self.model,
        )

    async def _handle_async_streaming_logging(
        self,
    ):
        """Handle the logging after all chunks have been collected (async)."""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        asyncio.create_task(
            PassThroughStreamingHandler._route_streaming_logging_to_handler(
                **self._build_logging_kwargs()
            )
        )

    def _handle_sync_streaming_logging(
        self,
    ):
        """
        Handle the logging after all chunks have been collected (sync).

        Sync callers were silently broken — ``__next__`` re-raised
        ``StopIteration`` without ever invoking the logging route, so every
        success_callback was skipped for sync iteration of the Google GenAI
        streaming API. This schedules the same async logging path the async
        iterator uses, finding (or creating) an event loop to run it on.
        """
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        coro = PassThroughStreamingHandler._route_streaming_logging_to_handler(
            **self._build_logging_kwargs()
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop in this thread; spin one up just to drain the coro.
            asyncio.run(coro)


class GoogleGenAIGenerateContentStreamingIterator(
    BaseGoogleGenAIGenerateContentStreamingIterator
):
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
        # Store the iterator once to avoid multiple stream consumption
        self.stream_iterator = response.iter_bytes()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Get the next chunk from the stored iterator
            chunk = next(self.stream_iterator)
            self.collected_chunks.append(chunk)
            # Just yield raw bytes
            return chunk
        except StopIteration:
            self._handle_sync_streaming_logging()
            raise

    def __aiter__(self):
        return self

    async def __anext__(self):
        # This should not be used for sync responses
        # If you need async iteration, use AsyncGoogleGenAIGenerateContentStreamingIterator
        raise NotImplementedError(
            "Use AsyncGoogleGenAIGenerateContentStreamingIterator for async iteration"
        )


class AsyncGoogleGenAIGenerateContentStreamingIterator(
    BaseGoogleGenAIGenerateContentStreamingIterator
):
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
        # Store the async iterator once to avoid multiple stream consumption
        self.stream_iterator = response.aiter_bytes()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Get the next chunk from the stored async iterator
            chunk = await self.stream_iterator.__anext__()
            self.collected_chunks.append(chunk)
            # Just yield raw bytes
            return chunk
        except StopAsyncIteration:
            await self._handle_async_streaming_logging()
            raise StopAsyncIteration
