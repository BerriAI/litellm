import base64
import json
from typing import Any, Optional

from litellm.litellm_core_utils.realtime_streaming import (
    RealTimeStreaming,
)  # Correct relative import
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging


class GeminiRealTimeStreaming(RealTimeStreaming):
    def __init__(
        self,
        websocket: Any,
        backend_ws: Any,
        model: str,  # Add model and config
        config: dict,
        logging_obj: Optional[LiteLLMLogging] = None,
    ):
        super().__init__(websocket, backend_ws, logging_obj)
        self.model = model
        self.config = config

    async def send_setup_message(self):
        """Sends the initial setup message required by the Gemini API."""
        setup_message = {
            "setup": {
                "model": self.model,
                "generation_config": self.config,
            }
        }
        await self.backend_ws.send(json.dumps(setup_message))

    async def backend_to_client_send_messages(self):
        import websockets

        try:
            while True:
                message = await self.backend_ws.recv()
                # --- MODIFICATION FOR GEMINI ---
                try:
                    response = json.loads(message)
                    server_content = response.get("serverContent")
                    if server_content:
                        model_turn = server_content.get("modelTurn")
                        if model_turn:
                            parts = model_turn.get("parts")
                            if parts:
                                for part in parts:
                                    if "inlineData" in part:
                                        pcm_data = base64.b64decode(
                                            part["inlineData"]["data"]
                                        )
                                        # Send the decoded audio data to the client.  How you do this
                                        # depends on your client.  You might send it as a base64 string,
                                        # or as raw bytes, or use a different mechanism entirely.
                                        await self.websocket.send_text(
                                            json.dumps(
                                                {
                                                    "audio_data": base64.b64encode(
                                                        pcm_data
                                                    ).decode("utf-8")
                                                }
                                            )
                                        )  # Example: Send as base64
                        turn_complete = server_content.get("turnComplete")
                        if turn_complete:
                            # Handle end of turn, if needed
                            pass
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
                    # Handle invalid JSON
                # --- END MODIFICATION ---

                # LOGGING
                self.store_message(message)
        except websockets.exceptions.ConnectionClosed:  # type: ignore
            pass
        except Exception as e:
            print(f"Error in backend_to_client: {e}")  # More robust error handling
        finally:
            await self.log_messages()
        import websockets

        try:
            while True:
                message = await self.websocket.receive_text()

                # --- MODIFICATION FOR GEMINI ---
                try:
                    user_input = json.loads(message)
                    text_input = user_input.get(
                        "text"
                    )  # Assuming the client sends {"text": "user input"}

                    if text_input:
                        msg = {
                            "client_content": {
                                "turns": [
                                    {"role": "user", "parts": [{"text": text_input}]}
                                ],
                                "turn_complete": True,
                            }
                        }
                        await self.backend_ws.send(json.dumps(msg))
                except json.JSONDecodeError:
                    print(f"Invalid JSON received from client: {message}")
                # --- END MODIFICATION ---

                # LOGGING
                self.store_input(message=message)
                # FORWARD TO BACKEND
                # await self.backend_ws.send(message) # Removed direct forwarding
        except websockets.exceptions.ConnectionClosed:  # type: ignore
            pass
        except Exception as e:
            print(f"Error in client_ack: {e}")
