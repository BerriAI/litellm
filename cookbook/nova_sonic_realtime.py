"""
Client script to test Nova Sonic realtime API through LiteLLM proxy.

This script connects to LiteLLM proxy's realtime endpoint and enables
speech-to-speech conversation with Bedrock Nova Sonic.

Prerequisites:
- LiteLLM proxy running with Bedrock configured
- pyaudio installed: pip install pyaudio
- websockets installed: pip install websockets

Usage:
    python nova_sonic_realtime.py
"""

import asyncio
import base64
import json
import os
import pyaudio
import websockets
from typing import Optional

# Bounded queue size for audio chunks (configurable via env to avoid unbounded memory)
AUDIO_QUEUE_MAXSIZE = int(os.getenv("LITELLM_ASYNCIO_QUEUE_MAXSIZE", 10_000))

# Audio configuration (matching Nova Sonic requirements)
INPUT_SAMPLE_RATE = 16000  # Nova Sonic expects 16kHz input
OUTPUT_SAMPLE_RATE = 24000  # Nova Sonic outputs 24kHz
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024

# LiteLLM proxy configuration
LITELLM_PROXY_URL = "ws://localhost:4000/v1/realtime?model=bedrock-sonic"
LITELLM_API_KEY = "sk-12345"  # Your LiteLLM API key


class RealtimeClient:
    """Client for LiteLLM realtime API with audio support."""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_active = False
        self.audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        self.pyaudio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

    async def connect(self):
        """Connect to LiteLLM proxy realtime endpoint."""
        print(f"Connecting to {self.url}...")
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.ws = await websockets.connect(
            self.url,
            additional_headers=headers,
            max_size=10 * 1024 * 1024,  # 10MB max message size
        )
        self.is_active = True
        print("✓ Connected to LiteLLM proxy")

    async def send_session_update(self):
        """Send session configuration."""
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": "You are a friendly assistant. Keep your responses short and conversational.",
                "voice": "matthew",
                "temperature": 0.8,
                "max_response_output_tokens": 1024,
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        }
        await self.ws.send(json.dumps(session_update))
        print("✓ Session configuration sent")

    async def receive_messages(self):
        """Receive and process messages from the server."""
        try:
            async for message in self.ws:
                if not self.is_active:
                    break

                try:
                    data = json.loads(message)
                    event_type = data.get("type")

                    if event_type == "session.created":
                        print(f"✓ Session created: {data.get('session', {}).get('id')}")

                    elif event_type == "response.created":
                        print("🤖 Assistant is responding...")

                    elif event_type == "response.text.delta":
                        # Print text transcription
                        delta = data.get("delta", "")
                        print(delta, end="", flush=True)

                    elif event_type == "response.audio.delta":
                        # Queue audio for playback
                        audio_b64 = data.get("delta", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await self.audio_queue.put(audio_bytes)

                    elif event_type == "response.text.done":
                        print()  # New line after text

                    elif event_type == "response.done":
                        print("✓ Response complete")

                    elif event_type == "error":
                        print(f"❌ Error: {data.get('error', {})}")

                    else:
                        # Debug: print other event types
                        print(f"[{event_type}]", end=" ")

                except json.JSONDecodeError:
                    print(f"Failed to parse message: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            print("\n✗ Connection closed")
        except Exception as e:
            print(f"\n✗ Error receiving messages: {e}")
        finally:
            self.is_active = False

    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send audio chunk to server."""
        if not self.is_active or not self.ws:
            return

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }
        await self.ws.send(json.dumps(message))

    async def commit_audio_buffer(self):
        """Commit the audio buffer to trigger processing."""
        if not self.is_active or not self.ws:
            return

        message = {"type": "input_audio_buffer.commit"}
        await self.ws.send(json.dumps(message))

    async def capture_audio(self):
        """Capture audio from microphone and send to server."""
        print("\n🎤 Starting audio capture...")
        print("Speak into your microphone. Press Ctrl+C to stop.\n")

        self.input_stream = self.pyaudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        try:
            while self.is_active:
                audio_data = self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
        except Exception as e:
            print(f"Error capturing audio: {e}")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()

    async def play_audio(self):
        """Play audio responses from the server."""
        print("🔊 Starting audio playback...")

        self.output_stream = self.pyaudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        try:
            while self.is_active:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=0.1
                    )
                    if audio_data:
                        self.output_stream.write(audio_data)
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()

    async def close(self):
        """Close the connection and cleanup."""
        self.is_active = False

        if self.ws:
            await self.ws.close()

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.pyaudio.terminate()
        print("\n✓ Connection closed")


async def main():
    """Main function to run the realtime client."""
    print("=" * 80)
    print("Bedrock Nova Sonic Realtime Client")
    print("=" * 80)
    print()

    client = RealtimeClient(LITELLM_PROXY_URL, LITELLM_API_KEY)

    try:
        # Connect to server
        await client.connect()

        # Send session configuration
        await client.send_session_update()

        # Wait a moment for session to be established
        await asyncio.sleep(0.5)

        # Start tasks
        receive_task = asyncio.create_task(client.receive_messages())
        capture_task = asyncio.create_task(client.capture_audio())
        playback_task = asyncio.create_task(client.play_audio())

        # Wait for user to interrupt
        await asyncio.gather(
            receive_task,
            capture_task,
            playback_task,
            return_exceptions=True,
        )

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


if __name__ == "__main__":
    print("\nMake sure:")
    print("1. LiteLLM proxy is running on port 4000")
    print("2. Bedrock is configured in proxy_server_config.yaml")
    print("3. AWS credentials are set")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
