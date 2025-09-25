# Vertex AI Live API WebSocket Passthrough

LiteLLM now supports WebSocket passthrough for the Vertex AI Live API, enabling real-time bidirectional communication with Gemini models.

## Overview

The Vertex AI Live API WebSocket passthrough allows you to:
- Connect to Vertex AI Live API through LiteLLM proxy
- Use existing Vertex AI authentication methods
- Pass through all WebSocket messages bidirectionally
- Support text, audio, video, and multimodal interactions
- Track costs automatically for all usage types

## Configuration

### Environment Variables

Set the following environment variables for Vertex AI authentication:

```bash
# Required
DEFAULT_VERTEXAI_PROJECT=your-project-id
DEFAULT_VERTEXAI_LOCATION=us-central1

# Optional - use one of these for authentication
DEFAULT_GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# OR run: gcloud auth application-default login
```

### Configuration File

Alternatively, configure in your `config.yaml`:

```yaml
litellm_settings:
  default_vertex_config:
    vertex_project: "your-project-id"
    vertex_location: "us-central1"
    vertex_credentials: "os.environ/GOOGLE_APPLICATION_CREDENTIALS"
```

## Usage

### WebSocket Endpoints

- `ws://your-proxy-host/v1/vertex-ai/live`
- `ws://your-proxy-host/vertex-ai/live`

### Query Parameters

- `project_id` (optional): Google Cloud project ID (can be set in config)
- `location` (optional): Vertex AI location (can be set in config, default: us-central1)

### Example Connection

```javascript
// If project_id and location are set in config, you can connect without query params
const ws = new WebSocket('ws://localhost:4000/v1/vertex-ai/live');

// Or specify them explicitly
const ws = new WebSocket('ws://localhost:4000/v1/vertex-ai/live?project_id=your-project-id&location=us-central1');
```

## Cost Tracking

The WebSocket passthrough automatically tracks costs for all usage types based on the [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing#model-optimizer-pricing):

### Supported Cost Tracking

- **Text**: Character-based or token-based pricing depending on model
- **Audio**: Per-second pricing for audio input/output
- **Video**: Per-second pricing for video input
- **Images**: Per-image pricing for image input

### Cost Calculation

Costs are calculated using the same methods as other Vertex AI models in LiteLLM:
- Uses `cost_per_character` for Gemini models
- Uses `cost_per_token` for partner models (Claude, Llama, etc.)
- Includes audio, video, and image costs when applicable

### Cost Logging

Costs are automatically logged to:
- LiteLLM proxy logs
- Database (if configured)
- Spend tracking system
- Admin dashboard

Example log output:
```
Vertex AI Live WebSocket session cost: $0.001234 (input: $0.000800, output: $0.000434) tokens: 150, characters: 1200, duration: 45.2s
```

## API Reference

### Setup Message

Send this message first to initialize the session:

```json
{
  "setup": {
    "model": "projects/your-project-id/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview-04-09",
    "generation_config": {
      "response_modalities": ["TEXT"]
    }
  }
}
```

### Text Input

```json
{
  "client_content": {
    "turns": [
      {
        "role": "user",
        "parts": [{"text": "Hello! How are you?"}]
      }
    ],
    "turn_complete": true
  }
}
```

### Audio Input

```json
{
  "realtime_input": {
    "media_chunks": [
      {
        "data": "base64-encoded-audio-data",
        "mime_type": "audio/pcm"
      }
    ]
  }
}
```

## Supported Features

### Response Modalities

- **TEXT**: Text responses
- **AUDIO**: Audio responses with voice synthesis

### Tools

- **Function Calling**: Define and use custom functions
- **Code Execution**: Execute Python code
- **Google Search**: Search the web
- **Voice Activity Detection**: Detect when user is speaking

### Advanced Features

- **Audio Transcription**: Transcribe input and output audio
- **Proactive Audio**: Model responds only when relevant
- **Affective Dialog**: Understand emotional expressions

## Examples

### Python Client

```python
import asyncio
import json
import websockets

async def chat_with_gemini():
    uri = "ws://localhost:4000/v1/vertex-ai/live?project_id=your-project-id"
    
    async with websockets.connect(uri) as websocket:
        # Setup
        setup = {
            "setup": {
                "model": "projects/your-project-id/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview-04-09",
                "generation_config": {"response_modalities": ["TEXT"]}
            }
        }
        await websocket.send(json.dumps(setup))
        
        # Wait for setup response
        response = await websocket.recv()
        print(f"Setup: {response}")
        
        # Send message
        message = {
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": "Hello!"}]}],
                "turn_complete": True
            }
        }
        await websocket.send(json.dumps(message))
        
        # Receive response
        async for response in websocket:
            print(f"Response: {response}")
            # Check if turn is complete
            data = json.loads(response)
            if data.get("serverContent", {}).get("turnComplete"):
                break

asyncio.run(chat_with_gemini())
```

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:4000/v1/vertex-ai/live?project_id=your-project-id');

ws.onopen = function() {
    // Send setup
    const setup = {
        setup: {
            model: "projects/your-project-id/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview-04-09",
            generation_config: { response_modalities: ["TEXT"] }
        }
    };
    ws.send(JSON.stringify(setup));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    // Check if setup is complete
    if (data.setupComplete) {
        // Send a message
        const message = {
            client_content: {
                turns: [{ role: "user", parts: [{ text: "Hello!" }] }],
                turn_complete: true
            }
        };
        ws.send(JSON.stringify(message));
    }
};
```

## Error Handling

The WebSocket connection may close with these codes:

- `4001`: Vertex AI credentials not configured
- `4002`: Project ID not provided
- `1011`: Internal server error

## Authentication

The WebSocket passthrough uses the same authentication as other LiteLLM endpoints:

1. **API Key**: Pass `Authorization: Bearer your-api-key` header
2. **Vertex AI Credentials**: Set environment variables or config file

## Limitations

- Requires valid Google Cloud project with Vertex AI API enabled
- WebSocket connections are not persistent across server restarts
- Rate limits apply based on your Google Cloud quotas

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure Vertex AI credentials are properly configured
2. **Project Not Found**: Verify the project ID exists and has Vertex AI enabled
3. **Connection Refused**: Check that the LiteLLM proxy server is running

### Debug Mode

Enable debug logging to see detailed connection information:

```bash
export LITELLM_LOG=DEBUG
```

## Related Documentation

- [Vertex AI Live API Reference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-live)
- [LiteLLM Proxy Configuration](../proxy/)
- [Vertex AI Passthrough Endpoints](./vertex_ai.md)
