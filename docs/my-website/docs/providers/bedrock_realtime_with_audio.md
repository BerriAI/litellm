# Bedrock Realtime API

## Overview

Amazon Bedrock's Nova Sonic model supports real-time bidirectional audio streaming for voice conversations. This tutorial shows how to use it through LiteLLM Proxy.

## Setup

### 1. Configure LiteLLM Proxy

Create a `config.yaml` file:

```yaml
model_list:
  - model_name: "bedrock-sonic"
    litellm_params:
      model: bedrock/amazon.nova-sonic-v1:0
      aws_region_name: us-east-1  # or your preferred region
    model_info:
      mode: realtime
```

### 2. Start LiteLLM Proxy

```bash
litellm --config config.yaml
```

## Basic Text Interaction

```python
import asyncio
import websockets
import json

LITELLM_API_KEY = "sk-1234"  # Your LiteLLM API key
LITELLM_URL = 'ws://localhost:4000/v1/realtime?model=bedrock-sonic'

async def test_text_conversation():
    async with websockets.connect(
        LITELLM_URL,
        additional_headers={
            "Authorization": f"Bearer {LITELLM_API_KEY}"
        }
    ) as ws:
        # Wait for session.created
        response = await ws.recv()
        print(f"Connected: {json.loads(response)['type']}")
        
        # Configure session
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": "You are a helpful assistant.",
                "modalities": ["text"],
                "temperature": 0.8
            }
        }
        await ws.send(json.dumps(session_update))
        
        # Send a message
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello!"}]
            }
        }
        await ws.send(json.dumps(message))
        
        # Trigger response
        await ws.send(json.dumps({"type": "response.create"}))
        
        # Listen for response
        while True:
            response = await ws.recv()
            event = json.loads(response)
            
            if event['type'] == 'response.text.delta':
                print(event['delta'], end='', flush=True)
            elif event['type'] == 'response.done':
                print("\n✓ Complete")
                break

if __name__ == "__main__":
    asyncio.run(test_text_conversation())
```

## Audio Streaming with Voice Conversation

```python
import asyncio
import websockets
import json
import base64
import pyaudio

LITELLM_API_KEY = "sk-1234"
LITELLM_URL = 'ws://localhost:4000/v1/realtime?model=bedrock-sonic'

# Audio configuration
INPUT_RATE = 16000   # Nova Sonic expects 16kHz input
OUTPUT_RATE = 24000  # Nova Sonic outputs 24kHz
CHUNK = 1024

async def audio_conversation():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Input stream (microphone)
    input_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    # Output stream (speakers)
    output_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=OUTPUT_RATE,
        output=True,
        frames_per_buffer=CHUNK
    )
    
    async with websockets.connect(
        LITELLM_URL,
        additional_headers={"Authorization": f"Bearer {LITELLM_API_KEY}"}
    ) as ws:
        # Wait for session.created
        await ws.recv()
        print("✓ Connected")
        
        # Configure session with audio
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": "You are a friendly voice assistant.",
                "modalities": ["text", "audio"],
                "voice": "matthew",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }
        await ws.send(json.dumps(session_update))
        print("🎤 Speak into your microphone...")
        
        async def send_audio():
            """Capture and send audio from microphone"""
            while True:
                audio_data = input_stream.read(CHUNK, exception_on_overflow=False)
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))
                await asyncio.sleep(0.01)
        
        async def receive_audio():
            """Receive and play audio responses"""
            while True:
                response = await ws.recv()
                event = json.loads(response)
                
                if event['type'] == 'response.audio.delta':
                    audio_b64 = event.get('delta', '')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        output_stream.write(audio_bytes)
                
                elif event['type'] == 'response.text.delta':
                    print(event['delta'], end='', flush=True)
                
                elif event['type'] == 'response.done':
                    print("\n✓ Response complete")
        
        # Run both tasks concurrently
        await asyncio.gather(send_audio(), receive_audio())

if __name__ == "__main__":
    try:
        asyncio.run(audio_conversation())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
```

## Using Tools/Function Calling

```python
import asyncio
import websockets
import json
from datetime import datetime

LITELLM_API_KEY = "sk-1234"
LITELLM_URL = 'ws://localhost:4000/v1/realtime?model=bedrock-sonic'

# Define tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def get_weather(location: str) -> dict:
    """Simulated weather function"""
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny"
    }

async def conversation_with_tools():
    async with websockets.connect(
        LITELLM_URL,
        additional_headers={"Authorization": f"Bearer {LITELLM_API_KEY}"}
    ) as ws:
        # Wait for session.created
        await ws.recv()
        
        # Configure session with tools
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": "You are a helpful assistant with access to tools.",
                "modalities": ["text"],
                "tools": TOOLS
            }
        }
        await ws.send(json.dumps(session_update))
        
        # Send a message that requires a tool
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What's the weather in San Francisco?"}]
            }
        }
        await ws.send(json.dumps(message))
        await ws.send(json.dumps({"type": "response.create"}))
        
        # Handle responses and tool calls
        while True:
            response = await ws.recv()
            event = json.loads(response)
            
            if event['type'] == 'response.text.delta':
                print(event['delta'], end='', flush=True)
            
            elif event['type'] == 'response.function_call_arguments.done':
                # Execute the tool
                function_name = event['name']
                arguments = json.loads(event['arguments'])
                
                print(f"\n🔧 Calling {function_name}({arguments})")
                result = get_weather(**arguments)
                
                # Send tool result back
                tool_result = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": event['call_id'],
                        "output": json.dumps(result)
                    }
                }
                await ws.send(json.dumps(tool_result))
                await ws.send(json.dumps({"type": "response.create"}))
            
            elif event['type'] == 'response.done':
                print("\n✓ Complete")
                break

if __name__ == "__main__":
    asyncio.run(conversation_with_tools())
```

## Configuration Options

### Voice Options
Available voices: `matthew`, `joanna`, `ruth`, `stephen`, `gregory`, `amy`

### Audio Formats
- **Input**: 16kHz PCM16 (mono)
- **Output**: 24kHz PCM16 (mono)

### Modalities
- `["text"]` - Text only
- `["audio"]` - Audio only  
- `["text", "audio"]` - Both text and audio

## Example Test Scripts

Complete working examples are available in the LiteLLM repository:

- **Basic audio streaming**: `test_bedrock_realtime_client.py`
- **Simple text test**: `test_bedrock_realtime_simple.py`
- **Tool calling**: `test_bedrock_realtime_tools.py`

## Requirements

```bash
pip install litellm websockets pyaudio
```

## AWS Configuration

Ensure your AWS credentials are configured:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION_NAME=us-east-1
```

Or use AWS CLI configuration:

```bash
aws configure
```

## Troubleshooting

### Connection Issues
- Ensure LiteLLM proxy is running on the correct port
- Verify AWS credentials are properly configured
- Check that the Bedrock model is available in your region

### Audio Issues
- Verify PyAudio is properly installed
- Check microphone/speaker permissions
- Ensure correct sample rates (16kHz input, 24kHz output)

### Tool Calling Issues
- Ensure tools are properly defined in session.update
- Verify tool results are sent back with correct call_id
- Check that response.create is sent after tool result

## Related Resources

- [OpenAI Realtime API Documentation](https://platform.openai.com/docs/guides/realtime)
- [Amazon Bedrock Nova Sonic Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/nova-sonic.html)
- [LiteLLM Realtime API Documentation](/docs/realtime)
