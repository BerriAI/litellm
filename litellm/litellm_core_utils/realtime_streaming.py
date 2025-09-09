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
                    if isinstance(transformed_response, list):
                        for event in transformed_response:
                            event_str = json.dumps(event)
                            ## LOGGING
                            self.store_message(event_str)
                            await self.websocket.send_text(event_str)
                    else:
                        event_str = json.dumps(transformed_response)
                        ## LOGGING
                        self.store_message(event_str)
                        await self.websocket.send_text(event_str)

                else:
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
