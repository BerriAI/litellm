"""
Bidirectional WebSocket streaming for the OpenAI Responses API WebSocket mode.

Unlike the Realtime API streaming (which handles audio sessions, VAD, and
guardrail interception on transcription events), the Responses WebSocket
protocol is simpler:

  Client ──response.create──▸  Backend
  Client ◂──streaming events── Backend

The client sends ``response.create`` JSON messages.  The backend sends back
streaming response events (the same events used by the SSE transport, but
delivered as individual WebSocket text frames).

This module handles the bidirectional forwarding and logging.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    CLIENT_CONNECTION_CLASS = ClientConnection
else:
    CLIENT_CONNECTION_CLASS = Any

DefaultLoggedResponsesEventTypes = [
    "response.create",
    "response.created",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "error",
]


class ResponsesWebSocketStreaming:
    """Bidirectional forwarder for the Responses API WebSocket transport."""

    def __init__(
        self,
        websocket: Any,
        backend_ws: CLIENT_CONNECTION_CLASS,
        logging_obj: LiteLLMLogging,
        user_api_key_dict: Optional[Any] = None,
    ):
        self.websocket = websocket
        self.backend_ws = backend_ws
        self.logging_obj = logging_obj
        self.user_api_key_dict = user_api_key_dict
        self.messages: List[Dict] = []
        self.input_messages: List[Dict] = []
        self.logged_event_types = DefaultLoggedResponsesEventTypes

    def _should_store_message(self, message_obj: dict) -> bool:
        msg_type = message_obj.get("type")
        if msg_type and msg_type in self.logged_event_types:
            return True
        return False

    def store_backend_message(self, raw: str) -> None:
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return
        if self._should_store_message(obj):
            self.messages.append(obj)

    def store_client_message(self, raw: str) -> None:
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return
        self.input_messages.append(obj)
        if self.logging_obj:
            self.logging_obj.pre_call(input=obj, api_key="")

    async def log_messages(self) -> None:
        if self.logging_obj and self.messages:
            asyncio.create_task(
                self.logging_obj.async_success_handler(self.messages)
            )

    async def backend_to_client(self) -> None:
        """Forward messages from the OpenAI backend to the proxy client."""
        from websockets.exceptions import ConnectionClosed

        try:
            while True:
                try:
                    raw = await self.backend_ws.recv(decode=False)
                except TypeError:
                    raw = await self.backend_ws.recv()  # type: ignore[assignment]

                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                self.store_backend_message(raw)
                await self.websocket.send_text(raw)
        except ConnectionClosed:
            verbose_logger.debug(
                "Responses WebSocket: backend connection closed"
            )
        except Exception as e:
            verbose_logger.exception(
                "Responses WebSocket: error forwarding backend→client: %s", e
            )
        finally:
            await self.log_messages()

    async def client_to_backend(self) -> None:
        """Forward messages from the proxy client to the OpenAI backend."""
        try:
            while True:
                message = await self.websocket.receive_text()
                self.store_client_message(message)
                await self.backend_ws.send(message)
        except Exception as e:
            verbose_logger.debug(
                "Responses WebSocket: client connection ended: %s", e
            )

    async def bidirectional_forward(self) -> None:
        forward_task = asyncio.create_task(self.backend_to_client())
        try:
            await self.client_to_backend()
        except Exception:
            forward_task.cancel()
        finally:
            if not forward_task.done():
                forward_task.cancel()
                try:
                    await forward_task
                except asyncio.CancelledError:
                    pass
