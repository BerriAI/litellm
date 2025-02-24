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
import copy
import datetime
import json
from typing import Any, Dict, List, Optional, Union
import uuid

import litellm

from .litellm_logging import Logging as LiteLLMLogging

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
        backend_ws: Any,
        logging_obj: Optional[LiteLLMLogging] = None,
        user_api_key_dict: Optional[dict] = None,
    ):
        self.websocket = websocket
        self.backend_ws = backend_ws
        self.logging_obj = logging_obj
        self.user_api_key_dict = user_api_key_dict
        self.messages: List = []
        self.input_message: Dict = {}

        _logged_real_time_event_types = litellm.logged_real_time_event_types

        if _logged_real_time_event_types is None:
            _logged_real_time_event_types = DefaultLoggedRealTimeEventTypes
        self.logged_real_time_event_types = _logged_real_time_event_types

    def _should_store_message(self, message: Union[str, bytes]) -> bool:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        message_obj = json.loads(message)
        _msg_type = message_obj["type"]
        if self.logged_real_time_event_types == "*":
            return True
        if _msg_type in self.logged_real_time_event_types:
            return True
        return False

    def store_message(self, message: Union[str, bytes]):
        """Store message in list"""
        if self._should_store_message(message):
            self.messages.append(message)

    def store_input(self, message: dict):
        """Store input message"""
        self.input_message = message
        if self.logging_obj:
            self.logging_obj.pre_call(input=message, api_key="")

    def message_to_dict(self, message: Union[str, bytes]):
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        message_obj = json.loads(message)
        return message_obj

    async def save_message_response_cost(self, message: Union[str, bytes]):
        message = self.message_to_dict(message)
        if message.get("type") != "response.done":
            return

        from litellm.litellm_core_utils.openai_realtime_tracking import (
            OpenAIRealtimeCostTracking,
        )

        self.logging_obj.model_call_details["completion_start_time"] = (
            datetime.datetime.now(datetime.timezone.utc)
        )
        self.logging_obj.model_call_details["litellm_call_id"] = str(uuid.uuid4())
        realtime_cost_tracking = OpenAIRealtimeCostTracking(
            token=self.user_api_key_dict["user_api_key"],
            user_id=self.user_api_key_dict["user_api_key_user_id"],
            end_user_id=self.user_api_key_dict["user_api_key_end_user_id"],
            team_id=self.user_api_key_dict["user_api_key_team_id"],
            kwargs=self.logging_obj.model_call_details,
            response=copy.deepcopy(message),
            org_id=self.user_api_key_dict["user_api_key_org_id"],
        )
        await self.logging_obj.async_websocket_success_handler(realtime_cost_tracking)
        self.logging_obj.websocket_success_handler(realtime_cost_tracking)

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
                message = await self.backend_ws.recv()
                await self.websocket.send_text(message)

                ## LOGGING
                self.store_message(message)
                await self.save_message_response_cost(message)
        except websockets.exceptions.ConnectionClosed:  # type: ignore
            pass
        except Exception:
            pass
        finally:
            await self.log_messages()

    async def client_ack_messages(self):
        try:
            while True:
                message = await self.websocket.receive_text()
                ## LOGGING
                self.store_input(message=message)
                ## FORWARD TO BACKEND
                await self.backend_ws.send(message)
        except self.websockets.exceptions.ConnectionClosed:  # type: ignore
            pass

    async def bidirectional_forward(self):

        forward_task = asyncio.create_task(self.backend_to_client_send_messages())
        try:
            await self.client_ack_messages()
        except self.websockets.exceptions.ConnectionClosed:  # type: ignore
            forward_task.cancel()
        finally:
            if not forward_task.done():
                forward_task.cancel()
                try:
                    await forward_task
                except asyncio.CancelledError:
                    pass
