import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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

    def store_input(self, message: dict):
        """Store input message"""
        self.input_message = message
        if self.logging_obj:
            self.logging_obj.pre_call(input=message, api_key="")

    async def log_messages(self):
        """Log messages in list"""
        if self.logging_obj:
            ## ASYNC LOGGING
            # Create an event loop for the new thread
            asyncio.create_task(self.logging_obj.async_success_handler(self.messages))
            ## SYNC LOGGING
            executor.submit(self.logging_obj.success_handler(self.messages))

    def _has_realtime_guardrails(self) -> bool:
        """Return True if any callback is registered for realtime_input_transcription."""
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        return any(
            isinstance(cb, CustomGuardrail)
            and cb.should_run_guardrail(
                data={},
                event_type=GuardrailEventHooks.realtime_input_transcription,
            )
            for cb in litellm.callbacks
        )

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

        for callback in litellm.callbacks:
            if not isinstance(callback, CustomGuardrail):
                continue
            if (
                callback.should_run_guardrail(
                    data={"transcript": transcript},
                    event_type=GuardrailEventHooks.realtime_input_transcription,
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
                # Cancel any in-flight response before speaking the warning.
                # This handles the race where create_response fired before we could intercept.
                await self.backend_ws.send(json.dumps({"type": "response.cancel"}))
                # Ask OpenAI to speak the warning — TTS audio plays naturally in the client
                await self.backend_ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "modalities": ["text", "audio"],
                                "instructions": (
                                    f"Say exactly and only: \"{safe_msg}\". "
                                    "Do not add anything else."
                                ),
                            },
                        }
                    )
                )
                verbose_logger.warning(
                    "[realtime guardrail] BLOCKED transcript: %r",
                    transcript[:80],
                )
                return True
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
                    returned_object = self.provider_config.transform_realtime_response(
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
                    self.current_output_item_id = returned_object[
                        "current_output_item_id"
                    ]
                    self.current_response_id = returned_object["current_response_id"]
                    self.current_delta_chunks = returned_object["current_delta_chunks"]
                    self.current_conversation_id = returned_object[
                        "current_conversation_id"
                    ]
                    self.current_item_chunks = returned_object["current_item_chunks"]
                    self.current_delta_type = returned_object["current_delta_type"]
                    self.session_configuration_request = returned_object[
                        "session_configuration_request"
                    ]
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
                        ## LOGGING
                        self.store_message(event_str)
                        await self.websocket.send_text(event_str)

                else:
                    ## GUARDRAIL: intercept transcription events before triggering LLM
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
                            continue
                    except (json.JSONDecodeError, AttributeError):
                        pass
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
