import base64
import json
import asyncio
from typing import Any, Optional, Dict, Union, List

import websockets

from litellm.litellm_core_utils.realtime_streaming import (
    RealTimeStreaming,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

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

        # Track connection state manually
        self.client_ws_open = True
        self.backend_ws_open = True

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

        # Add tools to the setup payload if they exist
        if self.tools and len(self.tools) > 0:
            setup_payload["tools"] = self.tools

        setup_message = {"setup": setup_payload}

        print(f"Gemini Setup Message: {json.dumps(setup_message)}")
        await self.backend_ws.send(json.dumps(setup_message))
        print("Gemini setup message sent.")

    async def wait_for_setup_complete(self):
        """Waits for the setupComplete message from the Gemini backend."""
        try:
            response = await self.backend_ws.recv()
            print(f"Setup response: {response}")
            # Parse response to check if it's a valid setup completion
            if isinstance(response, str):
                response_data = json.loads(response)
                if "setupComplete" not in response_data:
                    print(f"WARNING: Unexpected setup response format: {response}")
            return True
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed while waiting for Gemini setup complete: {e}")
            await self.safely_close_websocket(self.websocket, code=e.code, reason=f"Backend connection closed during setup: {e.reason}")
            self.backend_ws_open = False
            return False
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON during setup: {e}")
            await self.safely_close_websocket(self.websocket, code=1011, reason="Invalid JSON received during setup")
            await self.safely_close_websocket(self.backend_ws, code=1011, reason="Invalid JSON received during setup")
            self.client_ws_open = False
            self.backend_ws_open = False
            return False
        except Exception as e:
            print(f"An unexpected error occurred during Gemini setup: {e}")
            await self.safely_close_websocket(self.websocket, code=1011, reason=f"Unexpected setup error: {e}")
            await self.safely_close_websocket(self.backend_ws, code=1011, reason=f"Unexpected setup error: {e}")
            self.client_ws_open = False
            self.backend_ws_open = False
            return False

    async def safely_close_websocket(self, ws, code=1000, reason="Normal closure"):
        """Safely close a websocket without relying on .closed attribute"""
        try:
            await ws.close(code=code, reason=reason)
        except Exception as e:
            print(f"Error closing websocket: {e}")
        finally:
            if ws == self.websocket:
                self.client_ws_open = False
            elif ws == self.backend_ws:
                self.backend_ws_open = False

    async def is_websocket_open(self, ws):
        """Check if websocket is open by trying a simple operation"""
        try:
            # For some websocket implementations, we can use an attribute
            if hasattr(ws, 'closed'):
                return not ws.closed

            # For others, a state check might work
            if hasattr(ws, 'state') and hasattr(ws.state, 'name'):
                return ws.state.name == 'OPEN'

            # Default: assume it's open if it's in our tracking variables
            return (ws == self.websocket and self.client_ws_open) or \
                   (ws == self.backend_ws and self.backend_ws_open)
        except Exception:
            # If we can't determine, assume it's closed for safety
            return False

    async def backend_to_client_send_messages(self):
        """Receives messages from Gemini, transforms them to LiteLLM format, and forwards to the client."""
        try:
            while self.backend_ws_open and self.client_ws_open:
                message = await self.backend_ws.recv()
                # Log the raw message received from Gemini for debugging
                print(f"Received raw from Gemini backend: {message}")

                # Store the original raw message for logging purposes
                self.store_message(message)

                transformed_message_str = None
                try:
                    if isinstance(message, str):
                        gemini_data = json.loads(message)

                        # --- Transformation Logic ---
                        # Assume Gemini response structure (adjust if different)
                        # Example: {"candidates": [{"content": {"role": "model", "parts": [{"text": "..."}]}}]}
                        extracted_text = ""
                        lite_llm_role = "assistant" # Default LiteLLM role for model output

                        candidates = gemini_data.get("candidates")
                        if isinstance(candidates, list) and len(candidates) > 0:
                            # Process the first candidate
                            candidate = candidates[0]
                            content = candidate.get("content")
                            if isinstance(content, dict):
                                # Map Gemini's 'model' role to LiteLLM's 'assistant' role
                                if content.get("role") == "model":
                                    lite_llm_role = "assistant"
                                else:
                                    # Handle other potential roles if needed, or keep default
                                    lite_llm_role = content.get("role", "assistant")

                                parts = content.get("parts")
                                if isinstance(parts, list) and len(parts) > 0:
                                    # Concatenate text from all parts (or just take the first?)
                                    # For simplicity, let's concatenate text parts.
                                    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part]
                                    extracted_text = "".join(text_parts)

                        # Add other potential extraction paths if Gemini's format varies
                        # For example, sometimes streaming responses might be simpler:
                        # elif "text" in gemini_data:
                        #     extracted_text = gemini_data["text"]
                        #     lite_llm_role = "assistant" # Assume model role for simple text

                        if extracted_text:
                            # Construct the LiteLLM standard message format using 'parts'
                            lite_llm_message = {
                                "role": lite_llm_role,
                                "parts": [{"text": extracted_text}]
                                # Alternatively, if your client prefers 'content':
                                # "content": extracted_text
                            }
                            transformed_message_str = json.dumps(lite_llm_message)
                        else:
                            # Handle non-content messages (e.g., metadata, finish reasons)
                            # Option 1: Forward them raw if the client needs them
                            # transformed_message_str = message
                            # Option 2: Skip forwarding non-content messages
                            print(f"No text content extracted from Gemini message, skipping forward: {message}")
                            continue # Skip to next message

                    elif isinstance(message, bytes):
                        # If Gemini sends bytes, decide how to handle (e.g., forward raw)
                        print("Received bytes from Gemini, forwarding directly.")
                        if await self.is_websocket_open(self.websocket):
                            await self.websocket.send_bytes(message)
                        continue # Skip JSON transformation for bytes

                    else:
                        print(f"Received unexpected message type from Gemini: {type(message)}")
                        continue # Skip unknown types

                except json.JSONDecodeError:
                    print(f"Failed to decode JSON from Gemini: {message}")
                    # Decide how to handle non-JSON messages (e.g., forward raw string)
                    if isinstance(message, str) and await self.is_websocket_open(self.websocket):
                        print("Forwarding non-JSON string message raw.")
                        await self.websocket.send_text(message)
                    continue # Skip processing if not JSON
                except Exception as e:
                    print(f"Error processing/transforming message from Gemini: {e}")
                    # Decide how to handle errors (e.g., skip message)
                    continue # Skip message on transformation error


                # Send the transformed message (if available) to the client
                if transformed_message_str and await self.is_websocket_open(self.websocket):
                    print(f"Sending transformed to client: {transformed_message_str}")
                    await self.websocket.send_text(transformed_message_str)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Gemini backend connection closed: {e.code} {e.reason}")
            self.backend_ws_open = False
            if await self.is_websocket_open(self.websocket):
                await self.safely_close_websocket(self.websocket, code=e.code, reason=e.reason)
        except Exception as e:
            print(f"Error receiving from Gemini backend or sending to client: {e}")
            if await self.is_websocket_open(self.websocket):
                await self.safely_close_websocket(self.websocket, code=1011, reason=f"Error forwarding message: {e}")
            if await self.is_websocket_open(self.backend_ws):
                await self.safely_close_websocket(self.backend_ws, code=1011, reason=f"Error forwarding message: {e}")
        finally:
            # Log accumulated messages if needed (self.log_messages() might need adjustment
            # if it relies on the stored messages being in a specific format)
            # await self.log_messages()
            print("Backend-to-client message forwarding stopped.")


    async def client_to_backend_send_messages(self):
        """Receives messages from the client (e.g., Twilio app), formats them
        correctly for Gemini, and forwards to the Gemini backend."""
        try:
            while self.client_ws_open and self.backend_ws_open:
                message_text = await self.websocket.receive_text()
                print(f"Received from client: {message_text}")

                # Parse the message to check if it has the correct format
                try:
                    message_data = json.loads(message_text)
                    final_message_to_send = None # Store the final formatted message dict

                    # --- START AUDIO HANDLING ---
                    # Check if the message is audio input from the client (Twilio script)
                    if "audio" in message_data and "type" in message_data and message_data["type"] == "input_audio_buffer.append":
                        # Construct the message using the realtimeInput format
                        # based on the provided AudioInputMessage example.

                        audio_data_base64 = message_data["audio"]

                        # Determine MIME type based on Twilio setup.
                        # twilio_example_litellm.py sets "input_audio_format": "g711_ulaw".
                        # The standard MIME type for G.711 µ-law is audio/mulaw or audio/pcmu.
                        # It typically runs at 8000 Hz.
                        sample_rate = 8000
                        # Use the format from the example: audio/pcm;rate=... or audio/mulaw
                        # Let's try the specific mulaw type first.
                        # mime_type = f"audio/pcm;rate={sample_rate}" # As per example
                        mime_type = "audio/mulaw" # Standard for G.711 µ-law

                        # Structure according to the RealtimeInput/MediaChunk model example
                        # Use the exact field names 'mimeType' and 'mediaChunks'
                        media_chunk = {
                            "mimeType": mime_type,
                            "data": audio_data_base64
                        }
                        realtime_input_payload = {
                            "mediaChunks": [media_chunk]
                        }
                        final_message_to_send = {"realtimeInput": realtime_input_payload}
                    # --- END AUDIO HANDLING ---

                    # --- START OTHER MESSAGE HANDLING (Text, Tools, etc.) ---
                    # Handle text/history messages potentially coming in {"contents": [...]} format
                    elif "contents" in message_data and isinstance(message_data.get("contents"), list):
                        # Adapt the incoming 'contents' list (assumed to be turns) to the 'clientContent' format
                        content_turns = message_data["contents"]
                        valid_turns = []
                        for turn in content_turns:
                            if isinstance(turn, dict) and "role" in turn and "parts" in turn:
                                valid_turns.append(turn)
                            else:
                                print(f"WARNING: Skipping invalid turn structure in 'contents': {turn}")

                        if valid_turns:
                            is_complete = message_data.get("turn_complete", True)
                            final_message_to_send = {"clientContent": {"turns": valid_turns, "turn_complete": is_complete}}
                        else:
                            print(f"WARNING: No valid turns found in 'contents', cannot send message.")
                            continue

                    # For tool response messages, assume client sends correct format
                    elif "toolResponse" in message_data:
                        final_message_to_send = message_data # Pass through directly

                    # Handle potential cleanup of unsupported fields if message wasn't reformatted
                    elif final_message_to_send is None:
                        # If it wasn't audio, text, or tool response, check for and remove common unsupported fields
                        # before potentially forwarding (or deciding not to forward)
                        unsupported_fields = ["type", "session"]
                        cleaned_data = {k: v for k, v in message_data.items() if k not in unsupported_fields}
                        if cleaned_data != message_data:
                            print(f"WARNING: Removed unsupported fields. Result: {cleaned_data}")
                        # Decide if this cleaned_data is a valid Gemini message (e.g., setup?)
                        # For now, let's assume if it wasn't handled above, it's not valid to send.
                        # If you need to handle other types like 'setup', add specific elif blocks.
                        print(f"WARNING: Message from client is not a recognized/handled format: {message_data}")
                        continue # Skip sending unrecognized formats
                    # --- END OTHER MESSAGE HANDLING ---

                    # Convert the final formatted message back to string for sending
                    if final_message_to_send:
                        message_to_send_str = json.dumps(final_message_to_send)
                    else:
                        # Should not happen if logic is correct, but as a fallback
                        print(f"ERROR: final_message_to_send is None after processing.")
                        continue

                except json.JSONDecodeError:
                    print(f"WARNING: Received non-JSON message from client, cannot process for Gemini: {message_text}")
                    continue # Skip non-JSON messages
                except Exception as e:
                    print(f"ERROR: Failed processing client message: {e}")
                    continue # Skip sending this message on processing error

                self.store_input(message=final_message_to_send)
                print(f"Sending to Gemini backend: {message_to_send_str}")
                await self.backend_ws.send(message_to_send_str)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Client connection closed: {e.code} {e.reason}")
            self.client_ws_open = False
            if await self.is_websocket_open(self.backend_ws):
                await self.safely_close_websocket(self.backend_ws, code=e.code, reason=e.reason)
        except Exception as e:
            print(f"Error receiving from client or sending to Gemini backend: {e}")
            if await self.is_websocket_open(self.websocket):
                await self.safely_close_websocket(self.websocket, code=1011, reason=f"Error forwarding message: {e}")
            if await self.is_websocket_open(self.backend_ws):
                await self.safely_close_websocket(self.backend_ws, code=1011, reason=f"Error forwarding message: {e}")
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
            if await self.is_websocket_open(self.websocket):
                await self.safely_close_websocket(self.websocket, code=e.code, reason=f"Peer connection closed: {e.reason}")
            if await self.is_websocket_open(self.backend_ws):
                await self.safely_close_websocket(self.backend_ws, code=e.code, reason=f"Peer connection closed: {e.reason}")
        except Exception as e:
            print(f"An unexpected error occurred in bidirectional_forward: {e}")
            if await self.is_websocket_open(self.websocket):
                await self.safely_close_websocket(self.websocket, code=1011, reason=f"Forwarding error: {e}")
            if await self.is_websocket_open(self.backend_ws):
                await self.safely_close_websocket(self.backend_ws, code=1011, reason=f"Forwarding error: {e}")
        finally:
            if await self.is_websocket_open(self.websocket):
                print("Closing client websocket in finally block.")
                await self.safely_close_websocket(self.websocket)
            if await self.is_websocket_open(self.backend_ws):
                print("Closing backend websocket in finally block.")
                await self.safely_close_websocket(self.backend_ws)
            print("bidirectional_forward cleanup complete.")
