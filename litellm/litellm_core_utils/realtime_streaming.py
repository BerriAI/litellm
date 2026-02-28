import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamSessionEvents,
)
from litellm.types.realtime import ALL_DELTA_TYPES

from .litellm_logging import Logging as LiteLLMLogging

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    CLIENT_CONNECTION_CLASS = ClientConnection
else:
    CLIENT_CONNECTION_CLASS = Any

# Create a thread pool with a maximum of 10 threads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

DefaultLoggedRealTimeEventTypes = [
    "session.created",
    "response.create",
    "response.done",
]


class RealTimeStreaming:
    def __init__(
        self,
        websocket: Any,
        backend_ws: CLIENT_CONNECTION_CLASS,
        logging_obj: LiteLLMLogging,
        provider_config: Optional[BaseRealtimeConfig] = None,
        model: str = "",
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[Dict] = None,
    ):
        self.websocket = websocket
        self.backend_ws = backend_ws
        self.logging_obj = logging_obj
        self.messages: List[OpenAIRealtimeEvents] = []
        self.input_message: Dict = {}
        self.input_messages: List[Dict[str, str]] = []
        self.session_tools: List[Dict] = []
        self.tool_calls: List[Dict] = []

        _logged_real_time_event_types = litellm.logged_real_time_event_types

        if _logged_real_time_event_types is None:
            _logged_real_time_event_types = DefaultLoggedRealTimeEventTypes
        self.logged_real_time_event_types = _logged_real_time_event_types
        self.provider_config = provider_config
        self.model = model
        self.current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]] = None
        self.current_output_item_id: Optional[str] = None
        self.current_response_id: Optional[str] = None
        self.current_conversation_id: Optional[str] = None
        self.current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]] = None
        self.current_delta_type: Optional[ALL_DELTA_TYPES] = None
        self.session_configuration_request: Optional[str] = None
        self.user_api_key_dict = user_api_key_dict
        self.request_data: Dict = request_data or {}
        # Violation counter for end_session_after_n_fails support
        self._violation_count: int = 0
        # When a text message is blocked, hold the guardrail reason so the next
        # response.create can be rewritten to include the failure context.
        self._pending_guardrail_message: Optional[str] = None

    def _should_store_message(
        self,
        message_obj: Union[dict, OpenAIRealtimeEvents],
    ) -> bool:
        _msg_type = message_obj["type"] if "type" in message_obj else None
        if self.logged_real_time_event_types == "*":
            return True
        if _msg_type and _msg_type in self.logged_real_time_event_types:
            return True
        return False

    def store_message(self, message: Union[str, bytes, OpenAIRealtimeEvents]):
        """Store message in list"""
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        if isinstance(message, dict):
            message_obj = message
        else:
            message_obj = json.loads(message)
        self._collect_tool_calls_from_response_done(cast(dict, message_obj))
        try:
            if (
                not isinstance(message, dict)
                or message_obj.get("type") == "session.created"
                or message_obj.get("type") == "session.updated"
            ):
                message_obj = OpenAIRealtimeStreamSessionEvents(**message_obj)  # type: ignore
            elif not isinstance(message, dict):
                message_obj = OpenAIRealtimeStreamResponseBaseObject(**message_obj)  # type: ignore
        except Exception as e:
            verbose_logger.debug(f"Error parsing message for logging: {e}")
            raise e
        if self._should_store_message(message_obj):
            self.messages.append(message_obj)

    def _collect_user_input_from_client_event(
        self, message: Union[str, dict]
    ) -> None:
        """Extract user text content from client WebSocket events for spend logging."""
        try:
            if isinstance(message, str):
                msg_obj = json.loads(message)
            elif isinstance(message, dict):
                msg_obj = message
            else:
                return

            msg_type = msg_obj.get("type", "")

            if msg_type == "conversation.item.create":
                item = msg_obj.get("item", {})
                if item.get("role") == "user":
                    content_list = item.get("content", [])
                    for content in content_list:
                        if (
                            isinstance(content, dict)
                            and content.get("type") == "input_text"
                        ):
                            text = content.get("text", "")
                            if text:
                                self.input_messages.append(
                                    {"role": "user", "content": text}
                                )
            elif msg_type == "session.update":
                session = msg_obj.get("session", {})
                instructions = session.get("instructions", "")
                if instructions:
                    self.input_messages.append(
                        {"role": "system", "content": instructions}
                    )
                tools = session.get("tools")
                if tools and isinstance(tools, list):
                    self.session_tools = tools
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    def _collect_user_input_from_backend_event(
        self, event_obj: Union[dict, OpenAIRealtimeEvents]
    ) -> None:
        """Extract user voice transcription from backend events for spend logging."""
        try:
            event_type = event_obj.get("type", "")
            if (
                event_type
                == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = cast(str, event_obj.get("transcript", ""))
                if transcript:
                    self.input_messages.append(
                        {"role": "user", "content": transcript}
                    )
        except (AttributeError, TypeError):
            pass

    def _collect_tool_calls_from_response_done(
        self, event_obj: Union[dict, OpenAIRealtimeEvents]
    ) -> None:
        """Extract function_call items from response.done events for spend logging."""
        try:
            if event_obj.get("type") != "response.done":
                return
            response = cast(Dict[str, Any], event_obj.get("response", {}))
            for item in response.get("output", []):
                if item.get("type") == "function_call":
                    self.tool_calls.append(
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    )
        except (AttributeError, TypeError):
            pass

    def store_input(self, message: Union[str, dict]):
        """Store input message"""
        self.input_message = message if isinstance(message, dict) else {}
        self._collect_user_input_from_client_event(message)
        if self.logging_obj:
            self.logging_obj.pre_call(input=message, api_key="")

    async def log_messages(self):
        """Log messages in list"""
        if self.logging_obj:
            if self.input_messages:
                self.logging_obj.model_call_details["messages"] = (
                    self.input_messages
                )
            if self.session_tools or self.tool_calls:
                self.logging_obj.model_call_details[
                    "realtime_tools"
                ] = self.session_tools
                self.logging_obj.model_call_details[
                    "realtime_tool_calls"
                ] = self.tool_calls
            ## ASYNC LOGGING
            # Create an event loop for the new thread
            asyncio.create_task(self.logging_obj.async_success_handler(self.messages))
            ## SYNC LOGGING
            executor.submit(self.logging_obj.success_handler(self.messages))

    async def _send_to_backend(self, message: str) -> None:
        """Send a message to the backend WebSocket.

        If a provider_config is set the message is first passed through
        transform_realtime_request so that provider-specific translation
        (e.g. dropping session.update for Vertex AI) is applied even for
        guardrail-injected messages.
        """
        if self.provider_config:
            transformed = self.provider_config.transform_realtime_request(
                message, self.model, self.session_configuration_request
            )
            for msg in transformed:
                await self.backend_ws.send(msg)  # type: ignore[union-attr]
        else:
            await self.backend_ws.send(message)  # type: ignore[union-attr]

    def _has_realtime_guardrails(self) -> bool:
        """Return True if any callback is registered for realtime guardrail event types."""
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        _realtime_event_types = [
            GuardrailEventHooks.realtime_input_transcription,
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]
        return any(
            isinstance(cb, CustomGuardrail)
            and any(
                cb.should_run_guardrail(
                    data=self.request_data,
                    event_type=et,
                )
                for et in _realtime_event_types
            )
            for cb in litellm.callbacks
        )

    def _has_audio_transcription_guardrails(self) -> bool:
        """Return True if any callback needs to run on audio transcriptions (VAD path).

        When this returns True, we inject a session.update to disable the LLM's
        auto-response so the guardrail can gate it first.

        Must match the same hook criteria as run_realtime_guardrails() so that
        any guardrail that would actually check the transcript also disables
        auto-response before the transcript arrives.
        """
        return self._has_realtime_guardrails()

    async def run_realtime_guardrails(
        self,
        transcript: str,
        item_id: Optional[str] = None,
    ) -> bool:
        """
        Run registered guardrails on a completed speech transcription.

        Returns True if blocked (synthetic warning already sent to client).
        Returns False if clean (caller should send response.create to the backend).
        """
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        _realtime_event_types = [
            GuardrailEventHooks.realtime_input_transcription,
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]
        _check_data = {**self.request_data, "transcript": transcript}
        _already_run: set = set()

        for callback in litellm.callbacks:
            if not isinstance(callback, CustomGuardrail):
                continue
            if id(callback) in _already_run:
                continue
            if not any(
                callback.should_run_guardrail(data=_check_data, event_type=et)
                for et in _realtime_event_types
            ):
                continue
            _already_run.add(id(callback))
            try:
                await callback.apply_guardrail(
                    inputs={"texts": [transcript], "images": []},
                    request_data={"user_api_key_dict": self.user_api_key_dict},
                    input_type="request",
                )
            except Exception as e:
                # Re-raise unexpected errors (no status_code/detail = programming bug, not a block).
                # HTTPException and guardrail-raised exceptions have a status_code or detail attr.
                is_guardrail_block = hasattr(e, "status_code") or isinstance(e, ValueError)
                if not is_guardrail_block:
                    verbose_logger.exception(
                        "[realtime guardrail] unexpected error in apply_guardrail: %s", e
                    )
                    raise
                # Extract the human-readable error from the detail dict (HTTPException)
                # or fall back to str(e) for plain ValueError.
                detail = getattr(e, "detail", None)
                if isinstance(detail, dict):
                    safe_msg = detail.get("error") or str(e)
                elif detail is not None:
                    safe_msg = str(detail)
                else:
                    safe_msg = str(e) or "I'm sorry, that request was blocked by the content filter."

                # Use realtime_violation_message if configured; fall back to guardrail error text.
                error_msg = getattr(callback, "realtime_violation_message", None) or safe_msg

                # Cancel any in-progress LLM response (e.g. VAD auto-response).
                await self._send_to_backend(json.dumps({"type": "response.cancel"}))
                # Send the policy violation hint (shows as small gray status text in UI).
                await self.websocket.send_text(
                    json.dumps({
                        "type": "error",
                        "error": {
                            "type": "guardrail_violation",
                            "message": error_msg,
                            "code": "content_policy_violation",
                        },
                    })
                )
                # Ask the LLM to voice the exact guardrail message so the
                # user hears it as audio in voice sessions (not just text).
                guardrail_prompt = (
                    f"Say exactly the following message to the user, word for word, "
                    f"do not add anything else: {error_msg}"
                )
                await self._send_to_backend(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": guardrail_prompt}],
                    },
                }))
                await self._send_to_backend(
                    json.dumps({"type": "response.create"})
                )

                self._violation_count += 1
                end_session_after: Optional[int] = getattr(
                    callback, "end_session_after_n_fails", None
                )
                should_end = getattr(callback, "on_violation", None) == "end_session" or (
                    end_session_after is not None
                    and self._violation_count >= end_session_after
                )
                if should_end:
                    verbose_logger.warning(
                        "[realtime guardrail] ending session after violation %d",
                        self._violation_count,
                    )
                    await self.backend_ws.close()  # type: ignore[union-attr]

                verbose_logger.warning(
                    "[realtime guardrail] BLOCKED transcript (violation %d): %r",
                    self._violation_count,
                    transcript[:80],
                )
                return True
        return False

    async def _handle_provider_config_message(self, raw_response) -> None:
        """Process a backend message when a provider_config is set (transformed path)."""
        returned_object = self.provider_config.transform_realtime_response(  # type: ignore[union-attr]
            raw_response,
            self.model,
            self.logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": self.session_configuration_request,
                "current_output_item_id": self.current_output_item_id,
                "current_response_id": self.current_response_id,
                "current_delta_chunks": self.current_delta_chunks,
                "current_conversation_id": self.current_conversation_id,
                "current_item_chunks": self.current_item_chunks,
                "current_delta_type": self.current_delta_type,
            },
        )

        transformed_response = returned_object["response"]
        self.current_output_item_id = returned_object["current_output_item_id"]
        self.current_response_id = returned_object["current_response_id"]
        self.current_delta_chunks = returned_object["current_delta_chunks"]
        self.current_conversation_id = returned_object["current_conversation_id"]
        self.current_item_chunks = returned_object["current_item_chunks"]
        self.current_delta_type = returned_object["current_delta_type"]
        self.session_configuration_request = returned_object["session_configuration_request"]
        events = (
            transformed_response
            if isinstance(transformed_response, list)
            else [transformed_response]
        )
        for event in events:
            event_str = json.dumps(event)
            ## For audio/VAD guardrail path: forward session.created first, then inject.
            if (
                isinstance(event, dict)
                and event.get("type") == "session.created"
                and self._has_audio_transcription_guardrails()
            ):
                self.store_message(event_str)
                await self.websocket.send_text(event_str)
                await self._send_to_backend(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"turn_detection": {"create_response": False}},
                        }
                    )
                )
                continue
            ## GUARDRAIL: run on transcription events in provider_config path too
            if (
                isinstance(event, dict)
                and event.get("type")
                == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = event.get("transcript", "")
                self._collect_user_input_from_backend_event(cast(dict, event))
                self.store_message(event_str)
                await self.websocket.send_text(event_str)
                blocked = await self.run_realtime_guardrails(
                    cast(str, transcript), item_id=cast(Optional[str], event.get("item_id"))
                )
                if not blocked:
                    await self._send_to_backend(
                        json.dumps({"type": "response.create"})
                    )
                continue
            ## LOGGING
            self.store_message(event_str)
            await self.websocket.send_text(event_str)

    async def _handle_raw_backend_message(self, raw_response) -> bool:
        """Process a backend message without provider_config (raw path).

        Returns True if the caller should skip the default store+forward (i.e. continue the loop).
        """
        try:
            event_obj = json.loads(raw_response)

            # For audio/VAD guardrail path: once the session is ready, tell the backend
            # not to auto-respond after VAD detects end-of-speech.  We send the
            # session.created to the client FIRST so the client is always in sync, then
            # inject the session.update so a potential error from the backend doesn't
            # arrive before the client sees session.created.
            if (
                event_obj.get("type") == "session.created"
                and self._has_audio_transcription_guardrails()
            ):
                self.store_message(raw_response)
                await self.websocket.send_text(raw_response)
                await self._send_to_backend(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"turn_detection": {"create_response": False}},
                        }
                    )
                )
                return True

            if (
                event_obj.get("type")
                == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = event_obj.get("transcript", "")
                self._collect_user_input_from_backend_event(event_obj)
                ## LOGGING — must happen before continue below
                self.store_message(raw_response)
                # Forward transcript to client so user sees what they said
                await self.websocket.send_text(raw_response)
                blocked = await self.run_realtime_guardrails(
                    transcript,
                    item_id=event_obj.get("item_id"),
                )
                if not blocked:
                    # Clean — trigger LLM response
                    await self._send_to_backend(
                        json.dumps({"type": "response.create"})
                    )
                return True
        except (json.JSONDecodeError, AttributeError):
            pass
        return False

    async def backend_to_client_send_messages(self):
        import websockets

        try:
            while True:
                try:
                    raw_response = await self.backend_ws.recv(  # type: ignore[union-attr]
                        decode=False
                    )  # improves performance
                except TypeError:
                    raw_response = await self.backend_ws.recv()  # type: ignore[union-attr, assignment]

                if self.provider_config:
                    try:
                        await self._handle_provider_config_message(raw_response)
                    except Exception as e:
                        verbose_logger.exception(
                            f"Error processing backend message, skipping: {e}"
                        )
                        continue
                else:
                    handled = await self._handle_raw_backend_message(raw_response)
                    if handled:
                        continue
                    ## LOGGING
                    self.store_message(raw_response)
                    await self.websocket.send_text(raw_response)

        except websockets.exceptions.ConnectionClosed as e:  # type: ignore
            verbose_logger.exception(
                f"Connection closed in backend to client send messages - {e}"
            )
        except Exception as e:
            verbose_logger.exception(f"Error in backend to client send messages: {e}")
        finally:
            await self.log_messages()

    async def client_ack_messages(self):
        try:
            while True:
                message = await self.websocket.receive_text()

                ## GUARDRAIL: intercept conversation.item.create for text-based injection.
                try:
                    msg_obj = json.loads(message)
                    msg_type = msg_obj.get("type")

                    if msg_type == "conversation.item.create":
                        # Check user text messages for prompt injection
                        item = msg_obj.get("item", {})
                        if item.get("role") == "user":
                            content_list = item.get("content", [])
                            texts = [
                                c.get("text", "")
                                for c in content_list
                                if isinstance(c, dict) and c.get("type") == "input_text"
                            ]
                            combined_text = " ".join(texts)
                            if combined_text:
                                blocked = await self.run_realtime_guardrails(
                                    combined_text
                                )
                                if blocked:
                                    # Store the guardrail reason so the next response.create
                                    # (sent automatically by the client) is rewritten to
                                    # include it as response instructions.
                                    self._pending_guardrail_message = combined_text
                                    continue  # don't forward the original blocked message

                    if msg_type == "response.create" and self._pending_guardrail_message:
                        # The guardrail already sent the synthetic AI bubble — drop this
                        # response.create so OpenAI doesn't generate an additional response.
                        self._pending_guardrail_message = None
                        continue

                except (json.JSONDecodeError, AttributeError):
                    pass

                ## LOGGING
                self.store_input(message=message)
                ## FORWARD TO BACKEND
                if self.provider_config:
                    message = self.provider_config.transform_realtime_request(
                        message, self.model
                    )

                    for msg in message:
                        await self.backend_ws.send(msg)  # type: ignore[union-attr]
                else:
                    await self.backend_ws.send(message)  # type: ignore[union-attr]

        except Exception as e:
            verbose_logger.debug(f"Error in client ack messages: {e}")

    async def bidirectional_forward(self):
        forward_task = asyncio.create_task(self.backend_to_client_send_messages())
        try:
            await self.client_ack_messages()
        except self.websocket.exceptions.ConnectionClosed:  # type: ignore
            verbose_logger.debug("Connection closed")
            forward_task.cancel()
        finally:
            if not forward_task.done():
                forward_task.cancel()
                try:
                    await forward_task
                except asyncio.CancelledError:
                    pass
