import asyncio
import json
from datetime import datetime
from typing import Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesStreamingResponse,
)
from litellm.types.utils import ModelResponse, TextCompletionResponse
from litellm.utils import CustomStreamWrapper


class BaseAnthropicMessagesStreamingIterator:
    """
    Base class for streaming iterators that process responses from the Anthropic Messages API.

    This class contains shared logic for both synchronous and asynchronous iterators.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.anthropic_messages_provider_config = anthropic_messages_provider_config
        self.collected_chunks = []
        self.start_time = datetime.now()

    def _process_chunk(self, chunk: Optional[str]):
        """Process a single chunk of data from the stream"""
        if not chunk:
            return None

        # Handle SSE format (data: {...})
        chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
        if chunk is None:
            return None

        # Handle "[DONE]" marker
        if chunk == STREAM_SSE_DONE_STRING:
            self.finished = True
            return None

        try:
            # Parse the JSON chunk
            parsed_chunk = json.loads(chunk)

            # Format as AnthropicMessagesStreamingResponse
            if isinstance(parsed_chunk, dict):
                anthropic_messages_streaming_response = self.anthropic_messages_provider_config.transform_response_to_anthropic_messages_streaming_response(
                    model=self.model,
                    parsed_chunk=parsed_chunk,
                    logging_obj=self.logging_obj,
                )
                self.collected_chunks.append(anthropic_messages_streaming_response)
                return anthropic_messages_streaming_response

            return None
        except json.JSONDecodeError:
            # If we can't parse the chunk, continue
            return None

    def _handle_logging_completed_response(self):
        """Base implementation - should be overridden by subclasses"""
        pass

    def _build_complete_streaming_response(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw Anthropic chunks

        - Converts anthropic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        anthropic_model_response_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )
        all_openai_chunks = []
        for _chunk_str in self.collected_chunks:
            try:
                transformed_openai_chunk = (
                    anthropic_model_response_iterator.chunk_parser(chunk=_chunk_str)
                )
                if transformed_openai_chunk is not None:
                    all_openai_chunks.append(transformed_openai_chunk)

                verbose_logger.debug(
                    "all openai chunks= %s",
                    json.dumps(all_openai_chunks, indent=4, default=str),
                )
            except (StopIteration, StopAsyncIteration):
                break
        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks
        )
        return complete_streaming_response


class AnthropicMessagesStreamingIterator(BaseAnthropicMessagesStreamingIterator):
    """
    Async iterator for processing streaming responses from the Anthropic Messages API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        super().__init__(
            response, model, anthropic_messages_provider_config, logging_obj
        )
        self.stream_iterator = response.aiter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self) -> AnthropicMessagesStreamingResponse:
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = await self.stream_iterator.__anext__()
                except StopAsyncIteration:
                    self.finished = True
                    self._handle_logging_completed_response()
                    raise StopAsyncIteration

                result = self._process_chunk(chunk)

                if self.finished:
                    self._handle_logging_completed_response()
                    raise StopAsyncIteration
                elif result is not None:
                    return result
                # If result is None, continue the loop to get the next chunk

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            self._handle_logging_completed_response()
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in async context"""
        completed_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = self._build_complete_streaming_response(
            litellm_logging_obj=self.logging_obj,
            model=self.model,
        )
        asyncio.create_task(
            self.logging_obj.async_success_handler(
                result=completed_response,
                start_time=self.start_time,
                end_time=datetime.now(),
                cache_hit=None,
            )
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=completed_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )


class SyncAnthropicMessagesStreamingIterator(BaseAnthropicMessagesStreamingIterator):
    """
    Synchronous iterator for processing streaming responses from the Anthropic Messages API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        super().__init__(
            response, model, anthropic_messages_provider_config, logging_obj
        )
        self.stream_iterator = response.iter_lines()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = next(self.stream_iterator)
                except StopIteration:
                    self.finished = True
                    self._handle_logging_completed_response()
                    raise StopIteration

                result = self._process_chunk(chunk)

                if self.finished:
                    self._handle_logging_completed_response()
                    raise StopIteration
                elif result is not None:
                    return result
                # If result is None, continue the loop to get the next chunk

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            self._handle_logging_completed_response()
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in sync context"""
        completed_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = self._build_complete_streaming_response(
            litellm_logging_obj=self.logging_obj,
            model=self.model,
        )

        run_async_function(
            async_function=self.logging_obj.async_success_handler,
            result=completed_response,
            start_time=self.start_time,
            end_time=datetime.now(),
            cache_hit=None,
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=completed_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )
