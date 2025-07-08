import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

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
    ):
        self.litellm_logging_obj = litellm_logging_obj
        self.request_body = request_body
        self.start_time = datetime.now()
        self.collected_chunks: List[bytes] = []
        self.model = model

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
    ):
        super().__init__(
            litellm_logging_obj=logging_obj,
            request_body=request_body or {},
            model=model,
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
    ):
        super().__init__(
            litellm_logging_obj=logging_obj,
            request_body=request_body or {},
            model=model,
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