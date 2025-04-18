import asyncio
import json
from datetime import datetime
from typing import Optional

import httpx

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.utils import CustomStreamWrapper


class BaseResponsesAPIStreamingIterator:
    """
    Base class for streaming iterators that process responses from the Responses API.

    This class contains shared logic for both synchronous and asynchronous iterators.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.responses_api_provider_config = responses_api_provider_config
        self.completed_response: Optional[ResponsesAPIStreamingResponse] = None
        self.start_time = datetime.now()

    def _process_chunk(self, chunk):
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

            # Format as ResponsesAPIStreamingResponse
            if isinstance(parsed_chunk, dict):
                openai_responses_api_chunk = (
                    self.responses_api_provider_config.transform_streaming_response(
                        model=self.model,
                        parsed_chunk=parsed_chunk,
                        logging_obj=self.logging_obj,
                    )
                )
                # Store the completed response
                if (
                    openai_responses_api_chunk
                    and openai_responses_api_chunk.type
                    == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
                ):
                    self.completed_response = openai_responses_api_chunk
                    self._handle_logging_completed_response()

                return openai_responses_api_chunk

            return None
        except json.JSONDecodeError:
            # If we can't parse the chunk, continue
            return None

    def _handle_logging_completed_response(self):
        """Base implementation - should be overridden by subclasses"""
        pass


class ResponsesAPIStreamingIterator(BaseResponsesAPIStreamingIterator):
    """
    Async iterator for processing streaming responses from the Responses API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        super().__init__(response, model, responses_api_provider_config, logging_obj)
        self.stream_iterator = response.aiter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = await self.stream_iterator.__anext__()
                except StopAsyncIteration:
                    self.finished = True
                    raise StopAsyncIteration

                result = self._process_chunk(chunk)

                if self.finished:
                    raise StopAsyncIteration
                elif result is not None:
                    return result
                # If result is None, continue the loop to get the next chunk

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in async context"""
        asyncio.create_task(
            self.logging_obj.async_success_handler(
                result=self.completed_response,
                start_time=self.start_time,
                end_time=datetime.now(),
                cache_hit=None,
            )
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=self.completed_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )


class SyncResponsesAPIStreamingIterator(BaseResponsesAPIStreamingIterator):
    """
    Synchronous iterator for processing streaming responses from the Responses API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        super().__init__(response, model, responses_api_provider_config, logging_obj)
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
                    raise StopIteration

                result = self._process_chunk(chunk)

                if self.finished:
                    raise StopIteration
                elif result is not None:
                    return result
                # If result is None, continue the loop to get the next chunk

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in sync context"""
        run_async_function(
            async_function=self.logging_obj.async_success_handler,
            result=self.completed_response,
            start_time=self.start_time,
            end_time=datetime.now(),
            cache_hit=None,
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=self.completed_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )


class MockResponsesAPIStreamingIterator(BaseResponsesAPIStreamingIterator):
    """
    mock iterator - some models like o1-pro do not support streaming, we need to fake a stream
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        self.raw_http_response = response
        super().__init__(
            response=response,
            model=model,
            responses_api_provider_config=responses_api_provider_config,
            logging_obj=logging_obj,
        )
        self.is_done = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        transformed_response = (
            self.responses_api_provider_config.transform_response_api_response(
                model=self.model,
                raw_response=self.raw_http_response,
                logging_obj=self.logging_obj,
            )
        )
        return ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=transformed_response,
        )

    def __iter__(self):
        return self

    def __next__(self) -> ResponsesAPIStreamingResponse:
        if self.is_done:
            raise StopIteration
        self.is_done = True
        transformed_response = (
            self.responses_api_provider_config.transform_response_api_response(
                model=self.model,
                raw_response=self.raw_http_response,
                logging_obj=self.logging_obj,
            )
        )
        return ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=transformed_response,
        )
