import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

import litellm
from litellm.constants import (
    LITELLM_MAX_STREAMING_DURATION_SECONDS,
    STREAM_SSE_DONE_STRING,
)
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
    ResponsesAPIRequestParams,
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
        self._stream_created_time: float = time.time()

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

    def _check_max_streaming_duration(self) -> None:
        """Raise litellm.Timeout if the stream has exceeded LITELLM_MAX_STREAMING_DURATION_SECONDS."""
        if LITELLM_MAX_STREAMING_DURATION_SECONDS is None:
            return
        elapsed = time.time() - self._stream_created_time
        if elapsed > LITELLM_MAX_STREAMING_DURATION_SECONDS:
            raise litellm.Timeout(
                message=f"Stream exceeded max streaming duration of {LITELLM_MAX_STREAMING_DURATION_SECONDS}s (elapsed {elapsed:.1f}s)",
                model=self.model or "",
                llm_provider=self.custom_llm_provider or "",
            )

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

                # Wrap encrypted_content in streaming events (output_item.added, output_item.done)
                if (
                    self.litellm_metadata
                    and self.litellm_metadata.get("encrypted_content_affinity_enabled")
                ):
                    event_type = getattr(openai_responses_api_chunk, "type", None)
                    if event_type in (
                        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                    ):
                        item = getattr(openai_responses_api_chunk, "item", None)
                        if item:
                            encrypted_content = getattr(item, "encrypted_content", None)
                            if encrypted_content and isinstance(encrypted_content, str):
                                model_id = (
                                    self.litellm_metadata.get("model_info", {}).get("id")
                                    if self.litellm_metadata
                                    else None
                                )
                                if model_id:
                                    wrapped_content = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                                        encrypted_content, model_id
                                    )
                                    setattr(item, "encrypted_content", wrapped_content)

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
            self._check_max_streaming_duration()
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = await self.stream_iterator.__anext__()
                except StopAsyncIteration:
                    self.finished = True
                    raise StopAsyncIteration

                self._check_max_streaming_duration()
                result = self._process_chunk(chunk)

                if self.finished:
                    raise StopAsyncIteration
                elif result is not None:
                    # Await hook directly instead of run_async_function
                    # (which spawns a thread + event loop per call)
                    result = await self._call_post_streaming_deployment_hook(
                        chunk=result,
                    )
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
            self._check_max_streaming_duration()
            while True:
                # Get the next chunk from the stream
                try:
                    chunk = next(self.stream_iterator)
                except StopIteration:
                    self.finished = True
                    raise StopIteration

                self._check_max_streaming_duration()
                result = self._process_chunk(chunk)

                if self.finished:
                    raise StopIteration
                elif result is not None:
                    # Sync path: use run_async_function for the hook
                    result = run_async_function(
                        async_function=self._call_post_streaming_deployment_hook,
                        chunk=result,
                    )
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


# ---------------------------------------------------------------------------
# WebSocket mode streaming (bidirectional forwarding)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection as _WsClientConnection

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.thread_pool_executor import executor as _ws_executor

RESPONSES_WS_LOGGED_EVENT_TYPES = [
    "response.created",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "error",
]


class ResponsesWebSocketStreaming:
    """
    Manages bidirectional WebSocket forwarding for the Responses API
    WebSocket mode (wss://.../v1/responses).

    Unlike the Realtime API, the Responses API WebSocket mode:
    - Uses response.create as the client-to-server event
    - Streams back the same events as the HTTP streaming Responses API
    - Supports previous_response_id for incremental continuation
    - Supports generate: false for warmup
    - One response at a time per connection (sequential, no multiplexing)
    """

    def __init__(
        self,
        websocket: Any,
        backend_ws: Any,
        logging_obj: LiteLLMLoggingObj,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[Dict] = None,
    ):
        self.websocket = websocket
        self.backend_ws = backend_ws
        self.logging_obj = logging_obj
        self.user_api_key_dict = user_api_key_dict
        self.request_data: Dict = request_data or {}
        self.messages: list[Dict] = []
        self.input_messages: list[Dict[str, str]] = []

    def _should_store_event(self, event_obj: dict) -> bool:
        return event_obj.get("type") in RESPONSES_WS_LOGGED_EVENT_TYPES

    def _store_event(self, event: Any) -> None:
        if isinstance(event, bytes):
            event = event.decode("utf-8")
        if isinstance(event, str):
            try:
                event_obj = json.loads(event)
            except (json.JSONDecodeError, TypeError):
                return
        else:
            event_obj = event

        if self._should_store_event(event_obj):
            self.messages.append(event_obj)

    def _collect_input_from_client_event(self, message: Any) -> None:
        """Extract user input content from response.create for logging."""
        try:
            if isinstance(message, str):
                msg_obj = json.loads(message)
            elif isinstance(message, dict):
                msg_obj = message
            else:
                return

            if msg_obj.get("type") != "response.create":
                return

            input_items = msg_obj.get("input", [])
            if isinstance(input_items, str):
                self.input_messages.append({"role": "user", "content": input_items})
                return

            if isinstance(input_items, list):
                for item in input_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "message" and item.get("role") == "user":
                        content = item.get("content", [])
                        if isinstance(content, str):
                            self.input_messages.append(
                                {"role": "user", "content": content}
                            )
                        elif isinstance(content, list):
                            for c in content:
                                if (
                                    isinstance(c, dict)
                                    and c.get("type") == "input_text"
                                ):
                                    text = c.get("text", "")
                                    if text:
                                        self.input_messages.append(
                                            {"role": "user", "content": text}
                                        )
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    def _store_input(self, message: Any) -> None:
        self._collect_input_from_client_event(message)
        if self.logging_obj:
            self.logging_obj.pre_call(input=message, api_key="")

    async def _log_messages(self) -> None:
        if not self.logging_obj:
            return
        if self.input_messages:
            self.logging_obj.model_call_details["messages"] = self.input_messages
        if self.messages:
            asyncio.create_task(
                self.logging_obj.async_success_handler(self.messages)
            )
            _ws_executor.submit(self.logging_obj.success_handler, self.messages)

    async def backend_to_client(self) -> None:
        """Forward events from backend WebSocket to the client."""
        import websockets

        try:
            while True:
                try:
                    raw_response = await self.backend_ws.recv(decode=False)  # type: ignore[union-attr]
                except TypeError:
                    raw_response = await self.backend_ws.recv()  # type: ignore[union-attr, assignment]

                if isinstance(raw_response, bytes):
                    response_str = raw_response.decode("utf-8")
                else:
                    response_str = raw_response

                self._store_event(response_str)
                await self.websocket.send_text(response_str)

        except websockets.exceptions.ConnectionClosed as e:  # type: ignore
            verbose_logger.debug(
                "Responses WS backend connection closed: %s", e
            )
        except Exception as e:
            verbose_logger.exception(
                "Error in responses WS backend_to_client: %s", e
            )
        finally:
            await self._log_messages()

    async def client_to_backend(self) -> None:
        """Forward response.create events from client to backend."""
        try:
            while True:
                message = await self.websocket.receive_text()

                self._store_input(message)
                self._store_event(message)
                await self.backend_ws.send(message)  # type: ignore[union-attr]

        except Exception as e:
            verbose_logger.debug("Responses WS client_to_backend ended: %s", e)

    async def bidirectional_forward(self) -> None:
        """Run both forwarding directions concurrently."""
        forward_task = asyncio.create_task(self.backend_to_client())
        try:
            await self.client_to_backend()
        except Exception:
            pass
        finally:
            if not forward_task.done():
                forward_task.cancel()
                try:
                    await forward_task
                except asyncio.CancelledError:
                    pass
            try:
                await self.backend_ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Managed WebSocket mode (HTTP-backed, provider-agnostic)
# ---------------------------------------------------------------------------

_RESPONSE_CREATE_PARAMS: frozenset = (
    ResponsesAPIRequestParams.__required_keys__ | ResponsesAPIRequestParams.__optional_keys__
)

_MANAGED_WS_SKIP_KWARGS: frozenset = frozenset(
    {
        "litellm_logging_obj",  
        "litellm_call_id", 
        "aresponses",  
        "_aresponses_websocket",  
        "user_api_key_dict", 
    }
)


class ManagedResponsesWebSocketHandler:
    """
    Handles Responses API WebSocket mode for providers that do not expose a
    native ``wss://`` responses endpoint.

    Instead of proxying to a provider WebSocket, this handler:
    - Listens for ``response.create`` events from the client
    - Makes HTTP streaming calls via ``litellm.aresponses(stream=True)``
    - Serialises and forwards every streaming event back over the WebSocket
    - Supports ``previous_response_id`` for multi-turn conversations via
      in-memory session tracking (avoids async DB-write timing issues)
    - Supports sequential requests over a single persistent connection

    This makes every provider that LiteLLM can reach over HTTP available on
    the WebSocket transport without any provider-specific changes.
    """

    def __init__(
        self,
        websocket: Any,
        model: str,
        logging_obj: "LiteLLMLoggingObj",
        user_api_key_dict: Optional[Any] = None,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.websocket = websocket
        self.model = model
        self.logging_obj = logging_obj
        self.user_api_key_dict = user_api_key_dict
        self.litellm_metadata: Dict[str, Any] = litellm_metadata or {}
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.custom_llm_provider = custom_llm_provider
        # Carry through safe pass-through kwargs (e.g. extra_headers)
        self.extra_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k not in _MANAGED_WS_SKIP_KWARGS
        }
        # In-memory session history: response_id → full accumulated message list.
        # Keyed by the DECODED (pre-encoding) response ID from response.completed.
        # This avoids the async DB-write race condition where spend logs haven't
        # been committed yet when the next response.create arrives.
        self._session_history: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_chunk(chunk: Any) -> Optional[str]:
        """Serialize a streaming chunk to a JSON string for WebSocket transmission."""
        try:
            if hasattr(chunk, "model_dump_json"):
                return chunk.model_dump_json(exclude_none=True)
            if hasattr(chunk, "model_dump"):
                return json.dumps(chunk.model_dump(exclude_none=True), default=str)
            if isinstance(chunk, dict):
                return json.dumps(chunk, default=str)
            return json.dumps(str(chunk))
        except Exception as exc:
            verbose_logger.debug("ManagedResponsesWS: failed to serialize chunk: %s", exc)
            return None

    async def _send_error(self, message: str, error_type: str = "server_error") -> None:
        try:
            await self.websocket.send_text(
                json.dumps({"type": "error", "error": {"type": error_type, "message": message}})
            )
        except Exception:
            pass

    def _get_history_messages(self, previous_response_id: str) -> List[Dict[str, Any]]:
        """
        Return accumulated message history for *previous_response_id*.

        The key is the *decoded* response ID (the raw provider response ID before
        LiteLLM base64-encodes it into the ``resp_...`` format).
        """
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            previous_response_id
        )
        raw_id = decoded.get("response_id", previous_response_id)
        return list(self._session_history.get(raw_id, []))

    def _store_history(self, response_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Store the complete accumulated message history for *response_id*.

        Replaces any prior value — callers are responsible for passing the full
        history (prior turns + current input + new output).
        """
        self._session_history[response_id] = messages

    @staticmethod
    def _extract_response_id(completed_event: Dict[str, Any]) -> Optional[str]:
        """
        Pull the raw (decoded) response ID out of a ``response.completed`` event.
        Returns *None* if the event doesn't contain a usable ID.
        """
        resp_obj = completed_event.get("response", {})
        encoded_id: Optional[str] = resp_obj.get("id") if isinstance(resp_obj, dict) else None
        if not encoded_id:
            return None
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(encoded_id)
        return decoded.get("response_id", encoded_id)

    @staticmethod
    def _extract_output_messages(completed_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert the output items in a ``response.completed`` event into
        Responses API message dicts suitable for the next turn's ``input``.
        """
        resp_obj = completed_event.get("response", {})
        if not isinstance(resp_obj, dict):
            return []
        messages: List[Dict[str, Any]] = []
        for item in resp_obj.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            role = item.get("role", "assistant")
            if item_type == "message":
                content_parts = item.get("content") or []
                text_parts = [
                    p.get("text", "")
                    for p in content_parts
                    if isinstance(p, dict) and p.get("type") in ("output_text", "text")
                ]
                text = "".join(text_parts)
                if text:
                    messages.append({"type": "message", "role": role, "content": [{"type": "output_text", "text": text}]})
            elif item_type == "function_call":
                messages.append(item)
        return messages

    @staticmethod
    def _input_to_messages(input_val: Any) -> List[Dict[str, Any]]:
        """
        Normalise the ``input`` field of a ``response.create`` event to a list
        of Responses API message dicts.
        """
        if isinstance(input_val, str):
            return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": input_val}]}]
        if isinstance(input_val, list):
            return [item for item in input_val if isinstance(item, dict)]
        return []

    # ------------------------------------------------------------------
    # _process_response_create sub-methods
    # ------------------------------------------------------------------

    async def _parse_message(self, raw_message: str) -> Optional[Dict[str, Any]]:
        """Parse raw WS text; return the message dict or None (JSON error / ignored type)."""
        try:
            msg_obj = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send_error("Invalid JSON in response.create event", "invalid_request_error")
            return None
        if msg_obj.get("type") != "response.create":
            # Silently ignore non-response.create messages (e.g. warmup pings)
            return None
        return msg_obj

    @staticmethod
    def _build_base_call_kwargs(msg_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Responses API params from the event, handling both wire formats:
          Nested: {"type": "response.create", "response": {"input": [...], ...}}
          Flat:   {"type": "response.create", "input": [...], "model": "...", ...}
        """
        nested = msg_obj.get("response")
        response_params: Dict[str, Any] = (
            nested
            if isinstance(nested, dict) and nested
            else {k: v for k, v in msg_obj.items() if k != "type"}
        )
        return {
            param: response_params[param]
            for param in _RESPONSE_CREATE_PARAMS
            if param in response_params and response_params[param] is not None
        }

    def _apply_history(
        self,
        call_kwargs: Dict[str, Any],
        previous_response_id: Optional[str],
        current_messages: List[Dict[str, Any]],
        prior_history: List[Dict[str, Any]],
    ) -> None:
        """Prepend in-memory turn history, or fall back to DB-based reconstruction."""
        if not previous_response_id:
            return
        if prior_history:
            call_kwargs["input"] = prior_history + current_messages
            verbose_logger.debug(
                "ManagedResponsesWS: prepended %d history messages for previous_response_id=%s",
                len(prior_history),
                previous_response_id,
            )
        else:
            verbose_logger.debug(
                "ManagedResponsesWS: no in-memory history for previous_response_id=%s; "
                "falling back to DB-based session reconstruction",
                previous_response_id,
            )
            # Fall back to DB-based session reconstruction (may work for
            # cross-connection multi-turn when spend logs are committed)
            call_kwargs["previous_response_id"] = previous_response_id

    def _inject_credentials(
        self, call_kwargs: Dict[str, Any], event_model: Optional[str]
    ) -> None:
        """Inject connection-level credentials and metadata into call_kwargs."""
        if self.api_key is not None:
            call_kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            call_kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            call_kwargs["timeout"] = self.timeout
        # Only propagate custom_llm_provider when no per-request model override exists.
        # If the payload specifies a different model, let litellm re-resolve the
        # provider so we don't accidentally force the wrong backend.
        if self.custom_llm_provider is not None and not event_model:
            call_kwargs["custom_llm_provider"] = self.custom_llm_provider
        if self.litellm_metadata:
            call_kwargs["litellm_metadata"] = dict(self.litellm_metadata)

    @staticmethod
    def _update_proxy_request(call_kwargs: Dict[str, Any], model: str) -> None:
        """Update proxy_server_request body so spend logs record the full request."""
        proxy_server_request = (call_kwargs.get("litellm_metadata") or {}).get(
            "proxy_server_request"
        ) or {}
        if not isinstance(proxy_server_request, dict):
            return
        body = dict(proxy_server_request.get("body") or {})
        body["input"] = call_kwargs.get("input")
        body["store"] = call_kwargs.get("store")
        body["model"] = model
        for k in ("tools", "tool_choice", "instructions", "metadata"):
            if k in call_kwargs and call_kwargs[k] is not None:
                body[k] = call_kwargs[k]
        proxy_server_request = {**proxy_server_request, "body": body}
        if "litellm_metadata" not in call_kwargs:
            call_kwargs["litellm_metadata"] = {}
        call_kwargs["litellm_metadata"]["proxy_server_request"] = proxy_server_request
        call_kwargs.setdefault("litellm_params", {})
        call_kwargs["litellm_params"]["proxy_server_request"] = proxy_server_request

    async def _stream_and_forward(
        self, model: str, call_kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Stream ``litellm.aresponses`` and forward every chunk over the WebSocket.

        Captures the ``response.completed`` event type from the chunk object
        directly (before serialization) to avoid a redundant JSON round-trip on
        every chunk.  Returns the completed event dict, or ``None``.
        """
        completed_event: Optional[Dict[str, Any]] = None
        stream_response = await litellm.aresponses(model=model, **call_kwargs)
        async for chunk in stream_response:  # type: ignore[union-attr]
            if chunk is None:
                continue
            # Read type from the object before serializing to avoid double JSON parse
            chunk_type = getattr(chunk, "type", None) or (
                chunk.get("type") if isinstance(chunk, dict) else None
            )
            serialized = self._serialize_chunk(chunk)
            if serialized is None:
                continue
            if chunk_type == "response.completed" and completed_event is None:
                try:
                    completed_event = json.loads(serialized)
                except Exception:
                    pass
            try:
                await self.websocket.send_text(serialized)
            except Exception as send_exc:
                verbose_logger.debug(
                    "ManagedResponsesWS: error sending chunk to client: %s", send_exc
                )
                return completed_event  # Client disconnected
        return completed_event

    def _save_turn_history(
        self,
        completed_event: Optional[Dict[str, Any]],
        prior_history: List[Dict[str, Any]],
        current_messages: List[Dict[str, Any]],
    ) -> None:
        """Store this turn in in-memory history for future previous_response_id lookups."""
        if completed_event is None:
            return
        new_response_id = self._extract_response_id(completed_event)
        if not new_response_id:
            return
        output_msgs = self._extract_output_messages(completed_event)
        all_messages = prior_history + current_messages + output_msgs
        self._store_history(new_response_id, all_messages)
        verbose_logger.debug(
            "ManagedResponsesWS: stored %d messages for response_id=%s",
            len(all_messages),
            new_response_id,
        )

    # ------------------------------------------------------------------
    # Core request handler
    # ------------------------------------------------------------------

    async def _process_response_create(self, raw_message: str) -> None:
        """
        Parse one ``response.create`` event, call ``litellm.aresponses(stream=True)``,
        and forward every streaming event to the client.

        Multi-turn support via in-memory session history
        ------------------------------------------------
        When ``previous_response_id`` is present in the event:
        1. Look up the accumulated message history in ``self._session_history``
           (keyed by the decoded provider response ID).
        2. Prepend those messages to the current ``input`` so the model has full
           conversation context.
        3. After the stream completes, extract the new response ID and output
           messages from ``response.completed`` and store them in
           ``self._session_history`` for the next turn.

        This in-memory approach avoids the async DB-write race condition that
        occurs when spend logs haven't been committed by the time the second
        ``response.create`` arrives over the same WebSocket connection.
        """
        msg_obj = await self._parse_message(raw_message)
        if msg_obj is None:
            return

        call_kwargs = self._build_base_call_kwargs(msg_obj)
        call_kwargs["stream"] = True

        event_model: Optional[str] = call_kwargs.pop("model", None)
        model = event_model or self.model

        previous_response_id: Optional[str] = call_kwargs.pop("previous_response_id", None)
        current_messages = self._input_to_messages(call_kwargs.get("input"))

        # Fetch history once; reused in both _apply_history and _save_turn_history
        prior_history = (
            self._get_history_messages(previous_response_id)
            if previous_response_id
            else []
        )

        self._apply_history(call_kwargs, previous_response_id, current_messages, prior_history)
        self._inject_credentials(call_kwargs, event_model)
        self._update_proxy_request(call_kwargs, model)
        call_kwargs.update(self.extra_kwargs)

        try:
            completed_event = await self._stream_and_forward(model, call_kwargs)
        except Exception as exc:
            verbose_logger.exception(
                "ManagedResponsesWS: error processing response.create: %s", exc
            )
            await self._send_error(str(exc))
            return

        self._save_turn_history(completed_event, prior_history, current_messages)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main loop: accept ``response.create`` events sequentially and handle
        each one before waiting for the next message.
        """
        try:
            while True:
                try:
                    message = await self.websocket.receive_text()
                except Exception as exc:
                    verbose_logger.debug(
                        "ManagedResponsesWS: client disconnected: %s", exc
                    )
                    break

                await self._process_response_create(message)

        except Exception as exc:
            verbose_logger.exception("ManagedResponsesWS: unexpected error: %s", exc)
            await self._send_error(f"Internal server error: {exc}")
