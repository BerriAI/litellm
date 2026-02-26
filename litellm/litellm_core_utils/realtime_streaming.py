import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
        # Buffer for response.text.delta events pending output-guardrail check
        self._pending_output_text_events: List[str] = []
        # Violation counter for end_session_after_n_fails support
        self._violation_count: int = 0

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
        self._collect_tool_calls_from_response_done(message_obj)
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

    def _collect_user_input_from_backend_event(self, event_obj: Any) -> None:
        """Extract user voice transcription from backend events for spend logging."""
        try:
            event_type = event_obj.get("type", "")
            if (
                event_type
                == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = event_obj.get("transcript", "")
                if transcript:
                    self.input_messages.append(
                        {"role": "user", "content": transcript}
                    )
        except (AttributeError, TypeError):
            pass

    def _collect_tool_calls_from_response_done(
        self, event_obj: Any
    ) -> None:
        """Extract function_call items from response.done events for spend logging."""
        try:
            if event_obj.get("type") != "response.done":
                return
            response = event_obj.get("response", {})
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

    def _has_realtime_guardrails(self) -> bool:
        """Return True if any callback is registered for pre_call (runs across all modalities)."""
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        return any(
            isinstance(cb, CustomGuardrail)
            and cb.should_run_guardrail(
                data={},
                event_type=GuardrailEventHooks.pre_call,
            )
            for cb in litellm.callbacks
        )

    async def _handle_violation_action(
        self, callback: Any, safe_msg: str, end_session: bool = False
    ) -> None:
        """
        Speak the violation message to the user, then optionally close the session.

        end_session=True is set either when on_violation=='end_session' or when the
        session-level end_session_after_n_fails threshold has been reached.
        """
        spoken_msg = getattr(callback, "realtime_violation_message", None) or safe_msg
        await self.backend_ws.send(json.dumps({"type": "response.cancel"}))
        await self.backend_ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "instructions": (
                            f"Say exactly and only: \"{spoken_msg}\". "
                            "Do not add anything else."
                        ),
                    },
                }
            )
        )
        if end_session:
            verbose_logger.warning(
                "[realtime guardrail] ending session after violation"
            )
            await self.backend_ws.close()

    async def run_realtime_guardrails(
        self,
        transcript: str,
        item_id: Optional[str] = None,
    ) -> bool:
        """
        Run registered guardrails on a completed speech transcription.

        On each violation, increments the session violation counter.
        If end_session_after_n_fails is configured and the counter reaches that
        threshold, the session is terminated after speaking the violation message.

        Returns True if blocked (synthetic warning already sent to client).
        Returns False if clean (caller should send response.create to the backend).
        """
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        for callback in litellm.callbacks:
            if not isinstance(callback, CustomGuardrail):
                continue
            if (
                callback.should_run_guardrail(
                    data={"transcript": transcript},
                    event_type=GuardrailEventHooks.pre_call,
                )
                is not True
            ):
                continue

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

                self._violation_count += 1
                end_session_after: Optional[int] = getattr(
                    callback, "end_session_after_n_fails", None
                )
                should_end = (
                    getattr(callback, "on_violation", None) == "end_session"
                    or (
                        end_session_after is not None
                        and self._violation_count >= end_session_after
                    )
                )
                await self._handle_violation_action(callback, safe_msg, end_session=should_end)
                verbose_logger.warning(
                    "[realtime guardrail] BLOCKED transcript (violation %d): %r",
                    self._violation_count,
                    transcript[:80],
                )
                return True
        return False

    def _has_realtime_output_guardrails(self) -> bool:
        """Return True if any callback is registered for post_call (runs across all modalities)."""
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        return any(
            isinstance(cb, CustomGuardrail)
            and cb.should_run_guardrail(
                data={},
                event_type=GuardrailEventHooks.post_call,
            )
            for cb in litellm.callbacks
        )

    async def run_realtime_output_guardrails(
        self, text: str
    ) -> Tuple[bool, str]:
        """
        Run registered guardrails on completed response text.

        Returns (True, error_msg) if blocked, (False, "") if clean.
        """
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        for callback in litellm.callbacks:
            if not isinstance(callback, CustomGuardrail):
                continue
            if (
                callback.should_run_guardrail(
                    data={"text": text},
                    event_type=GuardrailEventHooks.post_call,
                )
                is not True
            ):
                continue
            try:
                await callback.apply_guardrail(
                    inputs={"texts": [text], "images": []},
                    request_data={"user_api_key_dict": self.user_api_key_dict},
                    input_type="response",
                )
            except Exception as e:
                is_guardrail_block = hasattr(e, "status_code") or isinstance(
                    e, ValueError
                )
                if not is_guardrail_block:
                    verbose_logger.exception(
                        "[realtime output guardrail] unexpected error: %s", e
                    )
                    raise
                detail = getattr(e, "detail", None)
                if isinstance(detail, dict):
                    safe_msg = detail.get("error") or str(e)
                elif detail is not None:
                    safe_msg = str(detail)
                else:
                    safe_msg = (
                        str(e) or "Response blocked by content filter."
                    )
                verbose_logger.warning(
                    "[realtime output guardrail] BLOCKED output text: %r",
                    text[:80],
                )
                return True, safe_msg
        return False, ""

    async def _send_output_text_done(
        self, event: Any, event_str: str
    ) -> None:
        """
        Handle a response.text.done event through the output guardrail.

        If blocked: discards buffered deltas, sends replacement error delta+done.
        If clean: flushes buffered deltas then forwards the done event.
        """
        full_text = event.get("text", "")
        blocked_out, error_msg = await self.run_realtime_output_guardrails(full_text)
        if blocked_out:
            self._pending_output_text_events.clear()
            error_delta_str = json.dumps(
                {
                    "type": "response.text.delta",
                    "delta": error_msg,
                    "content_index": event.get("content_index", 0),
                    "item_id": event.get("item_id", ""),
                    "output_index": event.get("output_index", 0),
                    "response_id": event.get("response_id", ""),
                }
            )
            error_done = dict(event)
            error_done["text"] = error_msg
            error_done_str = json.dumps(error_done)
            self.store_message(error_done_str)
            await self.websocket.send_text(error_delta_str)
            await self.websocket.send_text(error_done_str)
        else:
            for pending in self._pending_output_text_events:
                self.store_message(pending)
                await self.websocket.send_text(pending)
            self._pending_output_text_events.clear()
            self.store_message(event_str)
            await self.websocket.send_text(event_str)

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
            ## GUARDRAIL: inject create_response=false on session.created
            if isinstance(event, dict) and event.get("type") == "session.created":
                if self._has_realtime_guardrails():
                    await self.backend_ws.send(
                        json.dumps(
                            {
                                "type": "session.update",
                                "session": {
                                    "turn_detection": {
                                        "type": "server_vad",
                                        "create_response": False,
                                    }
                                },
                            }
                        )
                    )
        for event in events:
            event_str = json.dumps(event)
            ## GUARDRAIL: run on transcription events in provider_config path too
            if (
                isinstance(event, dict)
                and event.get("type")
                == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = event.get("transcript", "")
                self._collect_user_input_from_backend_event(event)
                self.store_message(event_str)
                await self.websocket.send_text(event_str)
                blocked = await self.run_realtime_guardrails(
                    transcript, item_id=event.get("item_id")
                )
                if not blocked:
                    await self.backend_ws.send(
                        json.dumps({"type": "response.create"})
                    )
                continue
            ## OUTPUT GUARDRAIL: buffer text deltas; check on text done
            if (
                isinstance(event, dict)
                and event.get("type") == "response.text.delta"
                and self._has_realtime_output_guardrails()
            ):
                self._pending_output_text_events.append(event_str)
                continue
            if isinstance(event, dict) and event.get("type") == "response.text.done":
                await self._send_output_text_done(event, event_str)
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
            if event_obj.get("type") == "session.created":
                # If any realtime guardrails are registered, proactively
                # set create_response=false so the LLM never auto-responds
                # before our guardrail has a chance to run.
                if self._has_realtime_guardrails():
                    await self.backend_ws.send(
                        json.dumps(
                            {
                                "type": "session.update",
                                "session": {
                                    "turn_detection": {
                                        "type": "server_vad",
                                        "create_response": False,
                                    }
                                },
                            }
                        )
                    )
                    verbose_logger.debug(
                        "[realtime guardrail] injected create_response=false into session"
                    )

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
                    await self.backend_ws.send(
                        json.dumps({"type": "response.create"})
                    )
                return True

            if event_obj.get("type") == "response.text.delta":
                has_guardrails = self._has_realtime_output_guardrails()
                verbose_logger.warning(
                    "[realtime output guardrail] response.text.delta — _has_realtime_output_guardrails=%s callbacks=%s",
                    has_guardrails,
                    [type(c).__name__ for c in litellm.callbacks],
                )
                if has_guardrails:
                    self._pending_output_text_events.append(raw_response)
                    return True

            if event_obj.get("type") == "response.text.done":
                verbose_logger.warning(
                    "[realtime output guardrail] response.text.done — text=%r _has_guardrails=%s",
                    event_obj.get("text", "")[:60],
                    self._has_realtime_output_guardrails(),
                )
                await self._send_output_text_done(event_obj, raw_response)
                return True

        except (json.JSONDecodeError, AttributeError):
            verbose_logger.debug("[realtime] skipped malformed backend message")
        return False

    async def backend_to_client_send_messages(self):
        import websockets

        try:
            while True:
                try:
                    raw_response = await self.backend_ws.recv(
                        decode=False
                    )  # improves performance
                except TypeError:
                    raw_response = await self.backend_ws.recv()  # type: ignore[assignment]

                if self.provider_config:
                    await self._handle_provider_config_message(raw_response)
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
                                    continue  # don't forward to backend

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
                        await self.backend_ws.send(msg)
                else:
                    await self.backend_ws.send(message)

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
