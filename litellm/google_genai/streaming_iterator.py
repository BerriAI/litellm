from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

if TYPE_CHECKING:
    from litellm.llms.base_llm.google_genai.transformation import (
        BaseGoogleGenAIGenerateContentConfig,
    )
else:
    BaseGoogleGenAIGenerateContentConfig = Any

class BaseGoogleGenAIGenerateContentStreamingIterator:
    """
    Base class for Google GenAI Generate Content streaming iterators that provides common logic
    for streaming response handling and logging.
    """

    def __init__(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
    ):
        self.litellm_logging_obj = litellm_logging_obj
        self.request_body = request_body
        self.start_time = datetime.now()

    async def _handle_streaming_logging(self, collected_chunks: List[bytes]):
        """Handle the logging after all chunks have been collected."""
        pass


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
        )
        self.response = response
        self.model = model
        self.generate_content_provider_config = generate_content_provider_config
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider
        # Store the iterator once to avoid multiple stream consumption
        self.stream_iterator = response.iter_lines()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Get the next chunk from the stored iterator
            chunk = next(self.stream_iterator)
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
        )
        self.response = response
        self.model = model
        self.generate_content_provider_config = generate_content_provider_config
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider
        # Store the async iterator once to avoid multiple stream consumption
        self.stream_iterator = response.aiter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Get the next chunk from the stored async iterator
            chunk = await self.stream_iterator.__anext__()
            # Just yield raw bytes
            return chunk
        except StopAsyncIteration:
            raise StopAsyncIteration 