"""
Streaming iterators for the Interactions API.

This module provides streaming iterators that properly stream SSE responses
from the Google Interactions API, similar to the responses API streaming iterator.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.interactions.long_decode import (
    HOP_FAILURES,
    continuation_token_of,
    hop_request,
    interaction_of,
    log_failed_hop,
    sum_usage,
    usage_of,
)
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines, iter_sse_lines
from litellm.types.interactions import (
    InteractionsAPIStreamingResponse,
)
from litellm.utils import CustomStreamWrapper

_INTERACTIONS_STREAMING_CHUNK_ADAPTER: Final = TypeAdapter(
    object, config=ConfigDict(strict=True, hide_input_in_errors=True)
)
_INTERACTIONS_STREAMING_EVENT: Final = TypeAdapter(dict[str, object])


class BaseInteractionsAPIStreamingIterator:
    """
    Base class for streaming iterators that process responses from the Interactions API.

    This class contains shared logic for both synchronous and asynchronous iterators.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str | None,
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: dict[str, Any] | None = None,
        custom_llm_provider: str | None = None,
        request_body: Mapping[str, object] | None = None,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.request_body = request_body
        self.hop_count = 1
        self.pending_hop: Mapping[str, object] | None = None
        self.held_completed: InteractionsAPIStreamingResponse | None = None
        self.last_hop_token: str | None = None
        self.hop_usages: tuple[Mapping[str, object], ...] = ()
        self.interactions_api_config = interactions_api_config
        self.completed_response: InteractionsAPIStreamingResponse | None = None
        self.start_time = logging_obj.start_time

        # set request kwargs
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider

        # set hidden params for response headers
        _api_base: Final = get_api_base(
            model=model or "",
            optional_params=self.logging_obj.model_call_details.get("litellm_params", {}),
        )
        _model_info: Final[dict] = litellm_metadata.get("model_info", {}) if litellm_metadata else {}
        self._hidden_params = {
            "model_id": _model_info.get("id", None),
            "api_base": _api_base,
        }
        self._hidden_params["additional_headers"] = process_response_headers(self.response.headers or {})

    def _process_chunk(self, chunk: str) -> InteractionsAPIStreamingResponse | None:
        """Process a single chunk of data from the stream."""
        if not chunk:
            return None

        # Handle SSE format (data: {...})
        stripped_chunk: Final = CustomStreamWrapper.strip_sse_data_from_chunk(chunk)
        if stripped_chunk is None:
            return None

        # Handle "[DONE]" marker
        if stripped_chunk == STREAM_SSE_DONE_STRING:
            self.finished = True
            return None

        try:
            # Parse the JSON chunk
            parsed_chunk: Final = _INTERACTIONS_STREAMING_CHUNK_ADAPTER.validate_python(json.loads(stripped_chunk))

            # Format as InteractionsAPIStreamingResponse
            if isinstance(parsed_chunk, dict):
                if self.logging_obj.completion_start_time is None:
                    self.logging_obj.update_completion_start_time(completion_start_time=datetime.now())
                event: Final = _INTERACTIONS_STREAMING_EVENT.validate_python(parsed_chunk)
                if self.hop_count > 1 and event.get("event_type") == "interaction.created":
                    return None
                streaming_response: Final = self.interactions_api_config.transform_streaming_response(
                    model=self.model,
                    parsed_chunk=dict(self._with_merged_usage(event)),
                    logging_obj=self.logging_obj,
                )
                if self._holds_for_next_hop(event, streaming_response):
                    return None

                # Store the completed response.
                # Legacy schema signals completion via status="completed".
                # New schema (Api-Revision: 2026-05-20) uses event_type="interaction.completed".
                # Remove the legacy check after June 8, 2026.
                if streaming_response and (
                    getattr(streaming_response, "status", None) == "completed"
                    or getattr(streaming_response, "event_type", None) == "interaction.completed"
                ):
                    self.completed_response = streaming_response
                    self._handle_logging_completed_response()

                return streaming_response

            return None
        except json.JSONDecodeError:
            # If we can't parse the chunk, continue
            verbose_logger.debug("Failed to parse streaming chunk: %s...", stripped_chunk[:200])
            return None

    def _with_merged_usage(self, event: Mapping[str, object]) -> Mapping[str, object]:
        """The final hop's interaction.completed carries usage summed over every hop, like a non-streaming decode."""
        interaction: Final = interaction_of(event)
        usage: Final = usage_of(interaction)
        if (
            not self.hop_usages
            or event.get("event_type") != "interaction.completed"
            or interaction is None
            or usage is None
        ):
            return event
        return {**event, "interaction": {**interaction, "usage": sum_usage((*self.hop_usages, usage))}}

    def _holds_for_next_hop(
        self, event: Mapping[str, object], streaming_response: InteractionsAPIStreamingResponse
    ) -> bool:
        """A hop ending in requires_continuation is held back, and the stream goes on with the next hop."""
        interaction: Final = interaction_of(event)
        if self.request_body is None or event.get("event_type") != "interaction.completed" or interaction is None:
            return False
        if (token := continuation_token_of(interaction)) is None or token == self.last_hop_token:
            self.held_completed = None
            return False
        usage: Final = usage_of(interaction)
        self.hop_usages = (*self.hop_usages, *((usage,) if usage is not None else ()))
        self.pending_hop = hop_request(self.request_body, interaction, token)
        self.held_completed = streaming_response
        self.last_hop_token = token
        return True

    def _release_held_completed(self) -> InteractionsAPIStreamingResponse | None:
        """A hop that could not be opened ends the stream with the held requires_continuation event instead."""
        held: Final = self.held_completed
        self.pending_hop = None
        self.held_completed = None
        if held is not None:
            self.completed_response = held
            self._handle_logging_completed_response()
        return held

    def _handle_logging_completed_response(self):
        """Base implementation - should be overridden by subclasses."""


class InteractionsAPIStreamingIterator(BaseInteractionsAPIStreamingIterator):
    """
    Async iterator for processing streaming responses from the Interactions API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str | None,
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: dict[str, Any] | None = None,
        custom_llm_provider: str | None = None,
        request_body: Mapping[str, object] | None = None,
        open_next_hop: Callable[[Mapping[str, object]], Awaitable[httpx.Response]] | None = None,
    ):
        super().__init__(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
            request_body=request_body if open_next_hop is not None else None,
        )
        self.open_next_hop = open_next_hop
        self.stream_iterator = aiter_sse_lines(response.aiter_text())

    async def _start_next_hop(self) -> InteractionsAPIStreamingResponse | None:
        if self.pending_hop is None or self.open_next_hop is None:
            return None
        try:
            response: Final = await self.open_next_hop(self.pending_hop)
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return self._release_held_completed()
        self.stream_iterator = aiter_sse_lines(response.aiter_text())
        self.hop_count += 1
        self.pending_hop = None
        return None

    def __aiter__(self):
        return self

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = await self.stream_iterator.__anext__()
                except StopAsyncIteration:
                    if self.pending_hop is not None:
                        held = await self._start_next_hop()
                        if held is not None:
                            return held
                        continue
                    self.finished = True
                    if (held := self._release_held_completed()) is not None:
                        return held
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

        logging_response: Final = copy.deepcopy(self.completed_response)

        asyncio.create_task(
            self.logging_obj.dispatch_success_handlers(
                logging_response,
                start_time=self.start_time,
                end_time=datetime.now(),
                cache_hit=None,
                prefer_async_handlers=True,
            )
        )


class SyncInteractionsAPIStreamingIterator(BaseInteractionsAPIStreamingIterator):
    """
    Synchronous iterator for processing streaming responses from the Interactions API.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str | None,
        interactions_api_config: BaseInteractionsAPIConfig,
        logging_obj: LiteLLMLoggingObj,
        litellm_metadata: dict[str, Any] | None = None,
        custom_llm_provider: str | None = None,
        request_body: Mapping[str, object] | None = None,
        open_next_hop: Callable[[Mapping[str, object]], httpx.Response] | None = None,
    ):
        super().__init__(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
            request_body=request_body if open_next_hop is not None else None,
        )
        self.open_next_hop = open_next_hop
        self.stream_iterator = iter_sse_lines(response.iter_text())

    def _start_next_hop(self) -> InteractionsAPIStreamingResponse | None:
        if self.pending_hop is None or self.open_next_hop is None:
            return None
        try:
            response: Final = self.open_next_hop(self.pending_hop)
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return self._release_held_completed()
        self.stream_iterator = iter_sse_lines(response.iter_text())
        self.hop_count += 1
        self.pending_hop = None
        return None

    def __iter__(self):
        return self

    def __next__(self) -> InteractionsAPIStreamingResponse:
        try:
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = next(self.stream_iterator)
                except StopIteration:
                    if self.pending_hop is not None:
                        held = self._start_next_hop()
                        if held is not None:
                            return held
                        continue
                    self.finished = True
                    if (held := self._release_held_completed()) is not None:
                        return held
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

        logging_response: Final = copy.deepcopy(self.completed_response)

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
