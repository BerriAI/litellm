import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# xAI Voice Agent (Realtime API)

xAI's Grok Voice Agent provides real-time voice conversation capabilities through WebSocket connections, enabling natural bidirectional audio interactions.

| Feature | Description | Comments |
| --- | --- | --- |
| LiteLLM AI Gateway | ✅ |  |
| LiteLLM Python SDK | ✅ | Full support via `litellm.realtime()` |

## Quick Start

### Supported Model

| Model | Context | Features |
|-------|---------|----------|
| `xai/grok-4-1-fast-non-reasoning` | 2M tokens | Voice conversation, Function calling, Vision, Audio, Web search, Caching |

**Note:** xAI Realtime API uses the non-reasoning variant for optimal real-time performance.

## Python SDK Usage

### Basic Realtime Connection

```python
import asyncio
from litellm import realtime

async def test_xai_realtime():
    """
    Test xAI Grok Voice Agent via LiteLLM SDK
    """
    # Initialize realtime connection
    ws = await realtime(
        model="xai/grok-4-1-fast-non-reasoning",
        api_key="your-xai-api-key",  # or set XAI_API_KEY env var
    )
    
    # Connection established, xAI sends "conversation.created" event
    print("Connected to xAI Grok Voice Agent")
    
    # Send a message
    await ws.send_text(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Hello! How are you?"
            }]
        }
    }))
    
    # Request a response
    await ws.send_text(json.dumps({
        "type": "response.create"
    }))
    
    # Listen for responses
    async for message in ws:
        data = json.loads(message)
        print(f"Received: {data['type']}")
        
        if data['type'] == 'response.done':
            break
    
    await ws.close()

# Run the async function
asyncio.run(test_xai_realtime())
```

### With Audio Input/Output

```python
import asyncio
import json
from litellm import realtime

async def xai_voice_conversation():
    """
    Voice conversation with xAI Grok Voice Agent
    """
    ws = await realtime(
        model="xai/grok-4-1-fast-non-reasoning",
        api_key="your-xai-api-key",
    )
    
    # Send audio data (base64 encoded PCM16 24kHz)
    await ws.send_text(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_audio",
                "audio": "base64_encoded_audio_data_here"
            }]
        }
    }))
    
    # Request response with audio
    await ws.send_text(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": "Please respond in a friendly tone."
        }
    }))
    
    # Process streaming audio response
    async for message in ws:
        data = json.loads(message)
        
        if data['type'] == 'response.audio.delta':
            # Handle audio chunks
            audio_chunk = data['delta']
            # Process audio_chunk (play it, save it, etc.)
            
        elif data['type'] == 'response.done':
            break
    
    await ws.close()

asyncio.run(xai_voice_conversation())
```

## LiteLLM Proxy (AI Gateway) Usage

Load balance across multiple xAI deployments or combine with other providers.

### 1. Add Model to Config

```yaml
model_list:
  - model_name: grok-voice-agent
    litellm_params:
      model: xai/grok-4-1-fast-non-reasoning
      api_key: os.environ/XAI_API_KEY
    model_info:
      mode: realtime

  # Optional: Add fallback to OpenAI
  - model_name: grok-voice-agent
    litellm_params:
      model: openai/gpt-4o-realtime-preview-2024-10-01
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime
```

### 2. Start Proxy

```bash
litellm --config /path/to/config.yaml 

# RUNNING on http://0.0.0.0:4000
```

### 3. Test Connection

#### Python Client

```python
import asyncio
import websockets
import json

async def test_proxy():
    url = "ws://0.0.0.0:4000/v1/realtime?model=grok-voice-agent"
    
    async with websockets.connect(
        url,
        extra_headers={
            "Authorization": "Bearer sk-1234",  # Your LiteLLM proxy key
            "OpenAI-Beta": "realtime=v1"
        }
    ) as ws:
        # Wait for conversation.created event from xAI
        message = await ws.recv()
        print(f"Connected: {message}")
        
        # Send a message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Hello from LiteLLM proxy!"
                }]
            }
        }))
        
        # Request response
        await ws.send(json.dumps({
            "type": "response.create"
        }))
        
        # Listen for response
        async for message in ws:
            data = json.loads(message)
            print(f"Event: {data['type']}")
            
            if data['type'] == 'response.done':
                break

asyncio.run(test_proxy())
```

#### Node.js Client

```javascript
// test.js - Run with: node test.js
const WebSocket = require("ws");

const url = "ws://0.0.0.0:4000/v1/realtime?model=grok-voice-agent";

const ws = new WebSocket(url, {
    headers: {
        "Authorization": "Bearer sk-1234",
        "OpenAI-Beta": "realtime=v1",
    },
});

ws.on("open", function open() {
    console.log("Connected to xAI via LiteLLM proxy");
    
    // Send a message
    ws.send(JSON.stringify({
        type: "conversation.item.create",
        item: {
            type: "message",
            role: "user",
            content: [{
                type: "input_text",
                text: "What's the weather like?"
            }]
        }
    }));
    
    // Request response
    ws.send(JSON.stringify({
        type: "response.create",
        response: {
            modalities: ["text"],
            instructions: "Please assist the user."
        }
    }));
});

ws.on("message", function incoming(message) {
    const data = JSON.parse(message.toString());
    console.log(`Event: ${data.type}`);
    
    if (data.type === 'response.done') {
        ws.close();
    }
});

ws.on("error", function handleError(error) {
    console.error("Error: ", error);
});
```

## Key Differences from OpenAI

xAI's Grok Voice Agent has some differences from OpenAI's Realtime API:

| Feature | xAI | OpenAI | LiteLLM Handling |
|---------|-----|--------|------------------|
| Initial Event | `conversation.created` | `session.created` | ⚠️ Passed through as-is |
| WebSocket URL | `wss://api.x.ai/v1/realtime` | `wss://api.openai.com/v1/realtime` | ✅ Auto-configured |
| Model | `grok-4-1-fast-non-reasoning` | `gpt-4o-realtime-preview` | ✅ Via model prefix |
| Audio Format | PCM16 24kHz mono | PCM16 24kHz mono | ✅ Compatible |
| Context Window | 2M tokens | 128K tokens | N/A |

**What LiteLLM Handles:**
- ✅ Automatic URL routing to correct provider
- ✅ Authentication headers (no `OpenAI-Beta` header for xAI)
- ✅ WebSocket connection management
- ✅ All other event types are compatible

**What You Need to Handle:**
- ⚠️ Initial event type difference (`conversation.created` vs `session.created`)

**Tip:** Make your client compatible with both event types:
```python
# Handle both providers
if event['type'] in ['session.created', 'conversation.created']:
    print("Connection established")
```

## Related Documentation

- [xAI Chat/Text Models](/docs/providers/xai)
- [LiteLLM Realtime API Overview](/docs/realtime)
- [xAI Official Documentation](https://docs.x.ai/docs)

## Support

For issues or questions:
- [LiteLLM GitHub Issues](https://github.com/BerriAI/litellm/issues)
- [xAI Documentation](https://docs.x.ai/docs)
