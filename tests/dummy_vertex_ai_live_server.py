"""
Dummy Vertex AI Live API WebSocket Server for Testing

This server mimics the Vertex AI Live API WebSocket behavior for testing
the LiteLLM integration without incurring actual API costs.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

import websockets
from websockets.server import WebSocketServerProtocol


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyVertexAILiveServer:
    """Dummy Vertex AI Live API WebSocket server for testing."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
    async def start(self):
        """Start the dummy WebSocket server."""
        logger.info(f"Starting dummy Vertex AI Live API server on {self.host}:{self.port}")
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        logger.info(f"Server started on ws://{self.host}:{self.port}")
        return self.server
    
    async def stop(self):
        """Stop the dummy WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Server stopped")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connections."""
        client_id = str(uuid.uuid4())
        logger.info(f"New connection from {websocket.remote_address} (ID: {client_id})")
        
        # Store session info
        self.sessions[client_id] = {
            "websocket": websocket,
            "model": None,
            "session_id": None,
            "conversation_id": str(uuid.uuid4()),
            "response_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            async for message in websocket:
                await self.handle_message(client_id, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for client {client_id}")
        except Exception as e:
            logger.error(f"Error handling connection {client_id}: {e}")
        finally:
            # Clean up session
            if client_id in self.sessions:
                del self.sessions[client_id]
    
    async def handle_message(self, client_id: str, message: str):
        """Handle incoming messages from clients."""
        try:
            data = json.loads(message)
            logger.info(f"Received message from {client_id}: {data}")
            
            # Handle different message types
            if "setup" in data:
                await self.handle_setup(client_id, data["setup"])
            elif "client_content" in data:
                await self.handle_client_content(client_id, data["client_content"])
            elif "realtime_input" in data:
                await self.handle_realtime_input(client_id, data["realtime_input"])
            else:
                logger.warning(f"Unknown message type from {client_id}: {data}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {client_id}: {e}")
            await self.send_error(client_id, "Invalid JSON", "INVALID_JSON")
        except Exception as e:
            logger.error(f"Error processing message from {client_id}: {e}")
            await self.send_error(client_id, str(e), "PROCESSING_ERROR")
    
    async def handle_setup(self, client_id: str, setup_data: Dict[str, Any]):
        """Handle session setup."""
        session = self.sessions[client_id]
        session["model"] = setup_data.get("model", "gemini-2.0-flash-live-preview-04-09")
        session["session_id"] = str(uuid.uuid4())
        
        # Send setup complete response
        response = {
            "setupComplete": {
                "sessionId": session["session_id"],
                "model": session["model"],
                "createdAt": int(datetime.now().timestamp())
            }
        }
        
        await self.send_message(client_id, response)
        logger.info(f"Setup complete for {client_id}")
    
    async def handle_client_content(self, client_id: str, content_data: Dict[str, Any]):
        """Handle client content (user input)."""
        session = self.sessions[client_id]
        
        # Extract text from turns
        text_content = ""
        if "turns" in content_data:
            for turn in content_data["turns"]:
                if "parts" in turn:
                    for part in turn["parts"]:
                        if "text" in part:
                            text_content += part["text"]
        
        logger.info(f"Processing user input from {client_id}: {text_content}")
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate mock response
        await self.generate_mock_response(client_id, text_content)
    
    async def handle_realtime_input(self, client_id: str, input_data: Dict[str, Any]):
        """Handle realtime input (audio)."""
        session = self.sessions[client_id]
        logger.info(f"Received realtime input from {client_id}")
        
        # Simulate audio processing
        await asyncio.sleep(0.2)
        
        # Generate mock audio response
        await self.generate_mock_audio_response(client_id)
    
    async def generate_mock_response(self, client_id: str, user_input: str):
        """Generate a mock text response."""
        session = self.sessions[client_id]
        
        # Create a simple mock response based on input
        if "hello" in user_input.lower():
            response_text = "Hello! How can I help you today?"
        elif "weather" in user_input.lower():
            response_text = "I'd be happy to help with weather information, but I'm a test server so I can't provide real weather data."
        elif "test" in user_input.lower():
            response_text = "This is a test response from the dummy Vertex AI Live API server!"
        else:
            response_text = f"I received your message: '{user_input}'. This is a mock response from the test server."
        
        # Send response in chunks to simulate streaming
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = {
                "modelTurn": {
                    "parts": [{"text": word + " "}],
                    "turnComplete": i == len(words) - 1
                }
            }
            await self.send_message(client_id, chunk)
            await asyncio.sleep(0.05)  # Small delay between chunks
        
        # Send generation complete
        complete_response = {
            "generationComplete": {
                "usageMetadata": {
                    "promptTokenCount": len(user_input.split()),
                    "candidatesTokenCount": len(response_text.split()),
                    "totalTokenCount": len(user_input.split()) + len(response_text.split())
                }
            }
        }
        await self.send_message(client_id, complete_response)
        
        # Send turn complete
        turn_complete = {
            "serverContent": {
                "turnComplete": {
                    "turnCompleteReason": "STOP"
                }
            }
        }
        await self.send_message(client_id, turn_complete)
    
    async def generate_mock_audio_response(self, client_id: str):
        """Generate a mock audio response."""
        session = self.sessions[client_id]
        
        # Send mock audio data (base64 encoded silence)
        audio_data = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="  # 1 second of silence
        
        audio_chunk = {
            "modelTurn": {
                "parts": [{
                    "inlineData": {
                        "mimeType": "audio/pcm",
                        "data": audio_data
                    }
                }],
                "turnComplete": True
            }
        }
        await self.send_message(client_id, audio_chunk)
        
        # Send generation complete
        complete_response = {
            "generationComplete": {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            }
        }
        await self.send_message(client_id, complete_response)
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        if client_id in self.sessions:
            websocket = self.sessions[client_id]["websocket"]
            try:
                await websocket.send(json.dumps(message))
                logger.debug(f"Sent to {client_id}: {message}")
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection closed for {client_id}")
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
    
    async def send_error(self, client_id: str, message: str, code: str):
        """Send an error message to a client."""
        error_response = {
            "error": {
                "message": message,
                "code": code
            }
        }
        await self.send_message(client_id, error_response)


async def main():
    """Main function to run the dummy server."""
    server = DummyVertexAILiveServer()
    
    try:
        await server.start()
        logger.info("Dummy Vertex AI Live API server is running...")
        logger.info("Press Ctrl+C to stop")
        
        # Keep the server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
