import asyncio
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

import litellm
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    update_response_metadata,
)
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
from litellm.types.utils import CallTypes
from litellm.utils import CustomStreamWrapper, async_post_call_success_deployment_hook


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
        request_data: Optional[Dict[str, Any]] = None,
        call_type: Optional[str] = None,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.finished = False
        self.responses_api_provider_config = responses_api_provider_config
        self.completed_response: Optional[ResponsesAPIStreamingResponse] = None
        self.start_time = getattr(logging_obj, "start_time", datetime.now())
        self._failure_handled = False  # Track if failure handler has been called

        # track request context for hooks
        self.litellm_metadata = litellm_metadata
        self.custom_llm_provider = custom_llm_provider
        self.request_data: Dict[str, Any] = request_data or {}
        self.call_type: Optional[str] = call_type

        # set hidden params for response headers (e.g., x-litellm-model-id)
        # This matches the stream wrapper in litellm/litellm_core_utils/streaming_handler.py
        _api_base = get_api_base(
            model=model or "",
            optional_params=self.logging_obj.model_call_details.get(
                "litellm_params", {}
            ),
        )
        _model_info: Dict = (
            litellm_metadata.get("model_info", {}) if litellm_metadata else {}
        )
        self._hidden_params = {
            "model_id": _model_info.get("id", None),
            "api_base": _api_base,
        }
        self._hidden_params["additional_headers"] = process_response_headers(
            self.response.headers or {}
        )  # GUARANTEE OPENAI HEADERS IN RESPONSE

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
                    response = (
                        ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                            responses_api_response=response_object,
                            litellm_metadata=self.litellm_metadata,
                            custom_llm_provider=self.custom_llm_provider,
                        )
                    )
                    setattr(openai_responses_api_chunk, "response", response)

                # Allow callbacks to modify chunk before returning
                openai_responses_api_chunk = run_async_function(
                    async_function=self._call_post_streaming_deployment_hook,
                    chunk=openai_responses_api_chunk,
                )

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
        except Exception as e:
            # Trigger failure hooks before re-raising
            # This ensures failures are logged even when _process_chunk is called directly
            self._handle_failure(e)
            raise

    def _handle_logging_completed_response(self):
        """Base implementation - should be overridden by subclasses"""
        pass

    async def _call_post_streaming_deployment_hook(self, chunk):
        """
        Allow callbacks to modify streaming chunks before returning (parity with chat).
        """
        try:
            # Align with chat pipeline: use logging_obj model_call_details + call_type
            typed_call_type: Optional[CallTypes] = None
            if self.call_type is not None:
                try:
                    typed_call_type = CallTypes(self.call_type)
                except ValueError:
                    typed_call_type = None
            if typed_call_type is None:
                try:
                    typed_call_type = CallTypes(getattr(self.logging_obj, "call_type", None))
                except Exception:
                    typed_call_type = None

            request_data = self.request_data or getattr(
                self.logging_obj, "model_call_details", {}
            )
            callbacks = getattr(litellm, "callbacks", None) or []
            hooks_ran = False
            for callback in callbacks:
                if hasattr(callback, "async_post_call_streaming_deployment_hook"):
                    hooks_ran = True
                    result = await callback.async_post_call_streaming_deployment_hook(
                        request_data=request_data,
                        response_chunk=chunk,
                        call_type=typed_call_type,
                    )
                    if result is not None:
                        chunk = result
            if hooks_ran:
                setattr(chunk, "_post_streaming_hooks_ran", True)
            return chunk
        except Exception:
            return chunk

    async def call_post_streaming_hooks_for_testing(self, chunk):
        """
        Helper to invoke streaming deployment hooks explicitly (used in tests).
        """
        return await self._call_post_streaming_deployment_hook(chunk)

    def _run_post_success_hooks(self, end_time: datetime):
        """
        Run post-call deployment hooks and update metadata similar to chat pipeline.
        """
        if self.completed_response is None:
            return

        request_payload: Dict[str, Any] = {}
        if isinstance(self.request_data, dict):
            request_payload.update(self.request_data)
        try:
            if hasattr(self.logging_obj, "model_call_details"):
                request_payload.update(self.logging_obj.model_call_details)
        except Exception:
            pass
        if "litellm_params" not in request_payload:
            try:
                request_payload["litellm_params"] = getattr(
                    self.logging_obj, "model_call_details", {}
                ).get("litellm_params", {})
            except Exception:
                request_payload["litellm_params"] = {}

        try:
            update_response_metadata(
                result=self.completed_response,
                logging_obj=self.logging_obj,
                model=self.model,
                kwargs=request_payload,
                start_time=self.start_time,
                end_time=end_time,
            )
        except Exception:
            # Non-blocking
            pass

        try:
            typed_call_type: Optional[CallTypes] = None
            if self.call_type is not None:
                try:
                    typed_call_type = CallTypes(self.call_type)
                except ValueError:
                    typed_call_type = None
        except Exception:
            typed_call_type = None
        if typed_call_type is None:
            try:
                typed_call_type = CallTypes.responses
            except Exception:
                typed_call_type = None

        try:
            # Call synchronously; async hook will be executed via asyncio.run in a new loop
            run_async_function(
                async_function=async_post_call_success_deployment_hook,
                request_data=request_payload,
                response=self.completed_response,
                call_type=typed_call_type,
            )
        except Exception:
            pass

    def _handle_failure(self, exception: Exception):
        """
        Trigger failure handlers before bubbling the exception.
        Only calls handlers once even if called multiple times.
        """
        # Prevent double-calling failure handlers
        if self._failure_handled:
            return
        self._failure_handled = True
        
        traceback_exception = traceback.format_exc()
        try:
            run_async_function(
                async_function=self.logging_obj.async_failure_handler,
                exception=exception,
                traceback_exception=traceback_exception,
                start_time=self.start_time,
                end_time=datetime.now(),
            )
        except Exception:
            pass

        try:
            executor.submit(
                self.logging_obj.failure_handler,
                exception,
                traceback_exception,
                self.start_time,
                datetime.now(),
            )
        except Exception:
            pass


async def call_post_streaming_hooks_for_testing(iterator, chunk):
    """
    Module-level helper for tests to ensure hooks can be invoked even if the iterator is wrapped.
    """
    hook_fn = getattr(iterator, "_call_post_streaming_deployment_hook", None)
    if hook_fn is None:
        return chunk
    return await hook_fn(chunk)


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
        request_data: Optional[Dict[str, Any]] = None,
        call_type: Optional[str] = None,
    ):
        super().__init__(
            response,
            model,
            responses_api_provider_config,
            logging_obj,
            litellm_metadata,
            custom_llm_provider,
            request_data,
            call_type,
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

        except StopAsyncIteration:
            # Normal end of stream - don't log as failure
            raise
        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            self._handle_failure(e)
            raise e
        except Exception as e:
            self.finished = True
            self._handle_failure(e)
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in async context"""
        # Create a copy for logging to avoid modifying the response object that will be returned to the user
        # The logging handlers may transform usage from Responses API format (input_tokens/output_tokens)
        # to chat completion format (prompt_tokens/completion_tokens) for internal logging
        # Use model_dump + model_validate instead of deepcopy to avoid pickle errors with
        # Pydantic ValidatorIterator when response contains tool_choice with allowed_tools (fixes #17192)
        logging_response = self.completed_response
        if self.completed_response is not None and hasattr(self.completed_response, 'model_dump'):
            try:
                logging_response = type(self.completed_response).model_validate(
                    self.completed_response.model_dump()
                )
            except Exception:
                # Fallback to original if serialization fails
                pass

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
        self._run_post_success_hooks(end_time=datetime.now())


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
        request_data: Optional[Dict[str, Any]] = None,
        call_type: Optional[str] = None,
    ):
        super().__init__(
            response,
            model,
            responses_api_provider_config,
            logging_obj,
            litellm_metadata,
            custom_llm_provider,
            request_data,
            call_type,
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

        except StopIteration:
            # Normal end of stream - don't log as failure
            raise
        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            self._handle_failure(e)
            raise e
        except Exception as e:
            self.finished = True
            self._handle_failure(e)
            raise e

    def _handle_logging_completed_response(self):
        """Handle logging for completed responses in sync context"""
        # Create a copy for logging to avoid modifying the response object that will be returned to the user
        # The logging handlers may transform usage from Responses API format (input_tokens/output_tokens)
        # to chat completion format (prompt_tokens/completion_tokens) for internal logging
        # Use model_dump + model_validate instead of deepcopy to avoid pickle errors with
        # Pydantic ValidatorIterator when response contains tool_choice with allowed_tools (fixes #17192)
        logging_response = self.completed_response
        if self.completed_response is not None and hasattr(self.completed_response, 'model_dump'):
            try:
                logging_response = type(self.completed_response).model_validate(
                    self.completed_response.model_dump()
                )
            except Exception:
                # Fallback to original if serialization fails
                pass

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
        self._run_post_success_hooks(end_time=datetime.now())


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
        request_data: Optional[Dict[str, Any]] = None,
        call_type: Optional[str] = None,
    ):
        super().__init__(
            response=response,
            model=model,
            responses_api_provider_config=responses_api_provider_config,
            logging_obj=logging_obj,
            litellm_metadata=litellm_metadata,
            custom_llm_provider=custom_llm_provider,
            request_data=request_data,
            call_type=call_type,
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

        # Add cost to usage object if include_cost_in_streaming_usage is True
        if litellm.include_cost_in_streaming_usage and logging_obj is not None:
            usage_obj: Optional[ResponseAPIUsage] = getattr(
                transformed, "usage", None
            )
            if usage_obj is not None:
                try:
                    cost: Optional[float] = logging_obj._response_cost_calculator(
                        result=transformed
                    )
                    if cost is not None:
                        setattr(usage_obj, "cost", cost)
                except Exception:
                    # If cost calculation fails, continue without cost
                    pass

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
