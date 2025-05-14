"""
async with websockets.connect(  # type: ignore
                url,
                extra_headers={
                    "api-key": api_key,  # type: ignore
                },
            ) as backend_ws:
                forward_task = asyncio.create_task(
                    forward_messages(websocket, backend_ws)
                )

                try:
                    while True:
                        message = await websocket.receive_text()
                        await backend_ws.send(message)
                except websockets.exceptions.ConnectionClosed:  # type: ignore
                    forward_task.cancel()
                finally:
                    if not forward_task.done():
                        forward_task.cancel()
                        try:
                            await forward_task
                        except asyncio.CancelledError:
                            pass
"""

import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamSessionEvents,
)

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
        logging_obj: Optional[LiteLLMLogging] = None,
        provider_config: Optional[BaseRealtimeConfig] = None,
        model: str = "",
    ):
        self.websocket = websocket
        self.backend_ws = backend_ws
        self.logging_obj = logging_obj
        self.messages: List[
            Union[
                OpenAIRealtimeStreamResponseBaseObject,
                OpenAIRealtimeStreamSessionEvents,
            ]
        ] = []
        self.input_message: Dict = {}

        _logged_real_time_event_types = litellm.logged_real_time_event_types

        if _logged_real_time_event_types is None:
            _logged_real_time_event_types = DefaultLoggedRealTimeEventTypes
        self.logged_real_time_event_types = _logged_real_time_event_types
        self.provider_config = provider_config
        self.model = model

    def _should_store_message(
        self,
        message_obj: Union[
            dict,
            OpenAIRealtimeStreamSessionEvents,
            OpenAIRealtimeStreamResponseBaseObject,
        ],
    ) -> bool:
        _msg_type = message_obj["type"]
        if self.logged_real_time_event_types == "*":
            return True
        if _msg_type in self.logged_real_time_event_types:
            return True
        return False

    def store_message(self, message: Union[str, bytes]):
        """Store message in list"""
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        message_obj = json.loads(message)
        try:
            if (
                message_obj.get("type") == "session.created"
                or message_obj.get("type") == "session.updated"
            ):
                message_obj = OpenAIRealtimeStreamSessionEvents(**message_obj)  # type: ignore
            else:
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

                if self.provider_config and isinstance(raw_response, str):
                    raw_response = self.provider_config.transform_realtime_response(
                        raw_response
                    )

                await self.websocket.send_text(raw_response)

                ## LOGGING
                self.store_message(raw_response)
        except websockets.exceptions.ConnectionClosed as e:  # type: ignore
            verbose_logger.debug(
                f"Connection closed in backend to client send messages - {e}"
            )
            raise e
        except Exception as e:
            verbose_logger.debug(f"Error in backend to client send messages: {e}")
            raise e
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
                    message = self.provider_config.transform_realtime_request(message)

                await self.backend_ws.send(message)
        except self.websocket.exceptions.ConnectionClosed:  # type: ignore
            verbose_logger.debug("Connection closed")
            pass
        except Exception as e:
            verbose_logger.debug(f"Error in client ack messages: {e}")

    async def bidirectional_forward(self):
        if (
            self.provider_config
            and self.provider_config.requires_session_configuration()
        ):
            session_configuration_request = (
                self.provider_config.session_configuration_request(self.model)
            )
            if session_configuration_request is None:
                raise ValueError(
                    "Session configuration request is None, but requires_session_configuration is True"
                )
            await self.backend_ws.send(session_configuration_request)
            try:
                verbose_logger.info(await self.backend_ws.recv(decode=False))
            except TypeError:
                verbose_logger.info(await self.backend_ws.recv())

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
