import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# xAI Voice Agent (Realtime API)

xAI's Grok Voice Agent provides real-time voice conversation capabilities through WebSocket connections, enabling natural bidirectional audio interactions.

| Feature | Description | Comments |
| --- | --- | --- |
| LiteLLM AI Gateway | ✅ | Connect a WebSocket client to the proxy `/v1/realtime` endpoint |
| LiteLLM Python SDK | ❌ | Realtime is served through the gateway, not a direct SDK call |

## Quick Start

### Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| `xai/grok-voice-think-fast-1.0` | Recommended | Flagship speech-to-speech voice model |
| `xai/grok-voice-fast-1.0` | Deprecated | Legacy voice model |
| `xai/grok-voice-latest` | Alias | Always points to the newest voice model (currently `grok-voice-think-fast-1.0`) |

These are dedicated full-duplex models built for real-time speech-to-speech conversation. They support function calling, web search, X search, collections search, remote MCP tools, and 20+ languages with automatic language detection. The examples below use `grok-voice-latest`, which always tracks the newest release; pin to a versioned name such as `grok-voice-think-fast-1.0` when you need stable behavior across releases.

## How LiteLLM Connects

LiteLLM serves xAI's Voice Agent through the AI Gateway, so realtime traffic goes over the proxy's OpenAI-compatible `/v1/realtime` WebSocket endpoint rather than a direct Python SDK call. You point any standard WebSocket client (Python `websockets`, Node `ws`, or the OpenAI SDK) at the gateway, and it forwards the session to `wss://api.x.ai/v1/realtime` with the correct model and authentication headers. The proxy setup and a runnable client are below.

## LiteLLM Proxy (AI Gateway) Usage

Load balance across multiple xAI deployments or combine with other providers.

### 1. Add Model to Config

```yaml
model_list:
  - model_name: grok-voice-agent
    litellm_params:
      model: xai/grok-voice-latest
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
        # First event from the server is session.created
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
| WebSocket URL | `wss://api.x.ai/v1/realtime` | `wss://api.openai.com/v1/realtime` | ✅ Auto-configured |
| Model | `grok-voice-latest` | `gpt-4o-realtime-preview` | ✅ Via model prefix |
| Audio Format | PCM (8-48kHz), μ-law, A-law | PCM16 24kHz mono | ✅ Compatible |

LiteLLM auto-configures the xAI endpoint, sets the authentication headers (it does not send the `OpenAI-Beta` header to xAI), and manages the WebSocket connection. Beyond that there is nothing xAI-specific to handle: the Voice Agent API is OpenAI-compatible and emits `session.created` on connect, just like OpenAI. Audio responses stream as `response.output_audio.delta`, with the matching transcript on `response.output_audio_transcript.delta`.

## Related Documentation

- [xAI Chat/Text Models](/docs/providers/xai)
- [LiteLLM Realtime API Overview](/docs/realtime)
- [xAI Official Documentation](https://docs.x.ai/docs)

## Support

For issues or questions:
- [LiteLLM GitHub Issues](https://github.com/BerriAI/litellm/issues)
- [xAI Documentation](https://docs.x.ai/docs)
