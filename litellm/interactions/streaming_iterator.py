"""
Streaming iterators for the Interactions API.

This module provides streaming iterators that properly stream SSE responses
from the Google Interactions API, similar to the responses API streaming iterator.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.types.interactions import (
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.utils import CustomStreamWrapper


class BaseInteractionsAPIStreamingIterator:
    """
    Base class for streaming iterators that process responses from the Interactions API.

    This class contains shared logic for both synchronous and asynchronous iterators.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: Optional[str],
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.interactions_api_config = interactions_api_config
        self.completed_response: Optional[InteractionsAPIStreamingResponse] = None
        self.start_time = datetime.now()

        # set request kwargs
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider

        # set hidden params for response headers
        _api_base = get_api_base(
            model=model or "",
            optional_params=self.logging_obj.model_call_details.get(
                "litellm_params", {}
            ),
        )
        _model_info: Dict = litellm_metadata.get("model_info", {}) if litellm_metadata else {}
        self._hidden_params = {
            "model_id": _model_info.get("id", None),
            "api_base": _api_base,
        }
        self._hidden_params["additional_headers"] = process_response_headers(
            self.response.headers or {}
        )

    def _process_chunk(self, chunk: str) -> Optional[InteractionsAPIStreamingResponse]:
        """Process a single chunk of data from the stream."""
        if not chunk:
            return None

        # Handle SSE format (data: {...})
        stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
        if stripped_chunk is None:
            return None

        # Handle "[DONE]" marker
        if stripped_chunk == STREAM_SSE_DONE_STRING:
            self.finished = True
            return None

        try:
            # Parse the JSON chunk
            parsed_chunk = json.loads(stripped_chunk)

            # Format as InteractionsAPIStreamingResponse
            if isinstance(parsed_chunk, dict):
                streaming_response = self.interactions_api_config.transform_streaming_response(
                    model=self.model,
                    parsed_chunk=parsed_chunk,
                    logging_obj=self.logging_obj,
                )

                # Store the completed response (check for status=completed)
                if (
                    streaming_response
                    and getattr(streaming_response, "status", None) == "completed"
                ):
                    self.completed_response = streaming_response
                    self._handle_logging_completed_response()

                return streaming_response

            return None
        except json.JSONDecodeError:
            # If we can't parse the chunk, continue
            verbose_logger.debug(f"Failed to parse streaming chunk: {stripped_chunk[:200]}...")
            return None

    def _handle_logging_completed_response(self):
        """Base implementation - should be overridden by subclasses."""
        pass


class InteractionsAPIStreamingIterator(BaseInteractionsAPIStreamingIterator):
    """
    Async iterator for processing streaming responses from the Interactions API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: Optional[str],
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        super().__init__(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
        )
        self.stream_iterator = response.aiter_lines()

    def __aiter__(self):
        return self

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
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
        """Handle logging for completed responses in async context."""
        import copy
        logging_response = copy.deepcopy(self.completed_response)

        asyncio.create_task(
            self.logging_obj.async_success_handler(
                result=logging_response,
                start_time=self.start_time,
                end_time=datetime.now(),
                cache_hit=None,
            )
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=logging_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )


class SyncInteractionsAPIStreamingIterator(BaseInteractionsAPIStreamingIterator):
    """
    Synchronous iterator for processing streaming responses from the Interactions API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: Optional[str],
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
    ):
        super().__init__(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
        )
        self.stream_iterator = response.iter_lines()

    def __iter__(self):
        return self

    def __next__(self) -> InteractionsAPIStreamingResponse:
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
        """Handle logging for completed responses in sync context."""
        import copy
        logging_response = copy.deepcopy(self.completed_response)

        run_async_function(
            async_function=self.logging_obj.async_success_handler,
            result=logging_response,
            start_time=self.start_time,
            end_time=datetime.now(),
            cache_hit=None,
        )

        executor.submit(
            self.logging_obj.success_handler,
            result=logging_response,
            cache_hit=None,
            start_time=self.start_time,
            end_time=datetime.now(),
        )

