import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

import litellm
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
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
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.responses_api_provider_config = responses_api_provider_config
        self.completed_response: Optional[ResponsesAPIStreamingResponse] = None
        self.start_time = datetime.now()

        # set request kwargs
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider

    def _process_chunk(self, chunk) -> Optional[ResponsesAPIStreamingResponse]:
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

                # if "response" in parsed_chunk, then encode litellm specific information like custom_llm_provider
                response_object = getattr(openai_responses_api_chunk, "response", None)
                if response_object:
                    response = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                        responses_api_response=response_object,
                        litellm_metadata=self.litellm_metadata,
                        custom_llm_provider=self.custom_llm_provider,
                    )
                    setattr(openai_responses_api_chunk, "response", response)

                # Store the completed response
                if (
                    openai_responses_api_chunk
                    and getattr(openai_responses_api_chunk, "type", None)
                    == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
                ):
                    self.completed_response = openai_responses_api_chunk
                    # Add cost to usage object if include_cost_in_streaming_usage is True
                    if (
                        litellm.include_cost_in_streaming_usage
                        and self.logging_obj is not None
                    ):
                        response_obj: Optional[ResponsesAPIResponse] = getattr(
                            openai_responses_api_chunk, "response", None
                        )
                        if response_obj:
                            usage_obj: Optional[ResponseAPIUsage] = getattr(
                                response_obj, "usage", None
                            )
                            if usage_obj is not None:
                                try:
                                    cost: Optional[float] = (
                                        self.logging_obj._response_cost_calculator(
                                            result=response_obj
                                        )
                                    )
                                    if cost is not None:
                                        setattr(usage_obj, "cost", cost)
                                except Exception:
                                    # If cost calculation fails, continue without cost
                                    pass

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
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        super().__init__(
            response,
            model,
            responses_api_provider_config,
            logging_obj,
            litellm_metadata,
            custom_llm_provider,
        )
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
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        super().__init__(
            response,
            model,
            responses_api_provider_config,
            logging_obj,
            litellm_metadata,
            custom_llm_provider,
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
    Mock iterator—fake a stream by slicing the full response text into
    5 char deltas, then emit a completed event.

    Models like o1-pro don't support streaming, so we fake it.
    """

    CHUNK_SIZE = 5

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        super().__init__(
            response=response,
            model=model,
            responses_api_provider_config=responses_api_provider_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
        )

        # one-time transform
        transformed = (
            self.responses_api_provider_config.transform_response_api_response(
                model=self.model,
                raw_response=response,
                logging_obj=logging_obj,
            )
        )
        full_text = self._collect_text(transformed)

        # build a list of 5‑char delta events
        deltas = [
            OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                delta=full_text[i : i + self.CHUNK_SIZE],
                item_id=transformed.id,
                output_index=0,
                content_index=0,
            )
            for i in range(0, len(full_text), self.CHUNK_SIZE)
        ]

        # append the completed event
        self._events = deltas + [
            ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=transformed,
            )
        ]
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        evt = self._events[self._idx]
        self._idx += 1
        return evt

    def __iter__(self):
        return self

    def __next__(self) -> ResponsesAPIStreamingResponse:
        if self._idx >= len(self._events):
            raise StopIteration
        evt = self._events[self._idx]
        self._idx += 1
        return evt

    def _collect_text(self, resp: ResponsesAPIResponse) -> str:
        out = ""
        for out_item in resp.output:
            item_type = getattr(out_item, "type", None)
            if item_type == "message":
                for c in getattr(out_item, "content", []):
                    out += c.text
        return out
