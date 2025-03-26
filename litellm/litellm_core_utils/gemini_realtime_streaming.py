import base64
import json
import asyncio
from typing import Any, Optional, Dict, Union

import websockets

from litellm.litellm_core_utils.realtime_streaming import (
    RealTimeStreaming,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
import litellm

class GeminiRealTimeStreaming(RealTimeStreaming):
    def __init__(
        self,
        websocket: Any,
        backend_ws: Any,
        model: str,
        config: dict,
        logging_obj: Optional[LiteLLMLogging] = None,
        vertex_location: Optional[str] = None,
        vertex_project: Optional[str] = None,
        system_instruction: Optional[Dict] = None,
        tools: Optional[list] = None,
    ):
        super().__init__(websocket, backend_ws, logging_obj)
        self.model_id = model
        self.config = config
        self.vertex_location = vertex_location
        self.vertex_project = vertex_project
        self.system_instruction = system_instruction
        self.tools = tools

        if self.vertex_project and self.vertex_location:
            self.model_resource_name = f"projects/{self.vertex_project}/locations/{self.vertex_location}/publishers/google/models/{self.model_id}"
        else:
            self.model_resource_name = self.model_id
            print(f"Warning: vertex_project or vertex_location not provided. Using model_id directly: {self.model_resource_name}")

    async def send_initial_setup(self):
        """Sends the initial setup message required by the Gemini API."""
        setup_payload: Dict[str, Any] = {
            "model": self.model_resource_name,
        }
        if self.config:
            setup_payload["generation_config"] = self.config
        if self.system_instruction:
            setup_payload["system_instruction"] = self.system_instruction

        setup_message = {"setup": setup_payload}

        print(f"Gemini Setup Message: {json.dumps(setup_message)}")
        await self.backend_ws.send(json.dumps(setup_message))
        print("Gemini setup message sent.")

    async def wait_for_setup_complete(self):
        """Waits for the setupComplete message from the Gemini backend."""
        try:
            await self.backend_ws.recv()
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed while waiting for Gemini setup complete: {e}")
            if not self.websocket.closed:
                await self.websocket.close(code=e.code, reason=f"Backend connection closed during setup: {e.reason}")
            return False
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON during setup: {e} - Message: {message_str}")
            await self.websocket.close(code=1011, reason="Invalid JSON received during setup")
            await self.backend_ws.close(code=1011, reason="Invalid JSON received during setup")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during Gemini setup: {e}")
            if not self.websocket.closed:
                await self.websocket.close(code=1011, reason=f"Unexpected setup error: {e}")
            if not self.backend_ws.closed:
                await self.backend_ws.close(code=1011, reason=f"Unexpected setup error: {e}")
            return False

    def store_message(self, message: Union[str, bytes]):
        """Store received message in list"""
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                print("Warning: Could not decode backend message as UTF-8. Storing as bytes representation.")
                message = repr(message)
        self.messages.append(message)

    def store_input(self, message: str):
        """Store input message sent from client to backend"""
        try:
            parsed_message = json.loads(message)
            self.input_message = parsed_message
            if self.logging_obj:
                self.logging_obj.pre_call(input=self.input_message, api_key="")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode client message as JSON for logging: {message}")
            self.input_message = {"raw_message": message}
            if self.logging_obj:
                self.logging_obj.pre_call(input=self.input_message, api_key="")
        except Exception as e:
            print(f"Error storing input message: {e}")
            self.input_message = {"error": str(e), "raw_message": message}

    async def backend_to_client_send_messages(self):
        """Forwards messages received from Gemini backend to the client."""
        try:
            while True:
                message = await self.backend_ws.recv()
                if isinstance(message, bytes):
                    await self.websocket.send_bytes(message)
                else:
                    await self.websocket.send_text(message)

                self.store_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Gemini backend connection closed: {e.code} {e.reason}")
            if not self.websocket.closed:
                await self.websocket.close(code=e.code, reason=e.reason)
        except Exception as e:
            print(f"Error receiving from Gemini backend or sending to client: {e}")
            if not self.websocket.closed:
                await self.websocket.close(code=1011, reason=f"Error forwarding message: {e}")
            if not self.backend_ws.closed:
                await self.backend_ws.close(code=1011, reason=f"Error forwarding message: {e}")
        finally:
            await self.log_messages()
            print("Backend-to-client message forwarding stopped.")

    async def client_to_backend_send_messages(self):
        """Forwards messages received from the client to the Gemini backend."""
        try:
            while True:
                message = await self.websocket.receive_text()
                self.store_input(message=message)

                await self.backend_ws.send(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Client connection closed: {e.code} {e.reason}")
            if not self.backend_ws.closed:
                await self.backend_ws.close(code=e.code, reason=e.reason)
        except Exception as e:
            print(f"Error receiving from client or sending to Gemini backend: {e}")
            if not self.websocket.closed:
                await self.websocket.close(code=1011, reason=f"Error forwarding message: {e}")
            if not self.backend_ws.closed:
                await self.backend_ws.close(code=1011, reason=f"Error forwarding message: {e}")
        finally:
            print("Client-to-backend message forwarding stopped.")

    async def bidirectional_forward(self):
        """Orchestrates the Gemini WebSocket session: setup and message forwarding."""
        try:
            await self.send_initial_setup()

            setup_ok = await self.wait_for_setup_complete()
            if not setup_ok:
                print("Gemini setup failed. Aborting bidirectional forward.")
                return

            print("Gemini setup successful. Starting bidirectional message forwarding.")

            client_to_backend_task = asyncio.create_task(self.client_to_backend_send_messages())
            backend_to_client_task = asyncio.create_task(self.backend_to_client_send_messages())

            done, pending = await asyncio.wait(
                [client_to_backend_task, backend_to_client_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            print("Bidirectional forwarding finished.")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"A connection closed unexpectedly during bidirectional forward setup or task management: {e}")
            if not self.websocket.closed:
                await self.websocket.close(code=e.code, reason=f"Peer connection closed: {e.reason}")
            if not self.backend_ws.closed:
                await self.backend_ws.close(code=e.code, reason=f"Peer connection closed: {e.reason}")
        except Exception as e:
            print(f"An unexpected error occurred in bidirectional_forward: {e}")
            if not self.websocket.closed:
                await self.websocket.close(code=1011, reason=f"Forwarding error: {e}")
            if not self.backend_ws.closed:
                await self.backend_ws.close(code=1011, reason=f"Forwarding error: {e}")
        finally:
            if not self.websocket.closed:
                print("Closing client websocket in finally block.")
                await self.websocket.close()
            if not self.backend_ws.closed:
                print("Closing backend websocket in finally block.")
                await self.backend_ws.close()
            print("bidirectional_forward cleanup complete.")
