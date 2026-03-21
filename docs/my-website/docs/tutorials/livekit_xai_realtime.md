import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LiveKit xAI Realtime Voice Agent

Use LiveKit's xAI Grok Voice Agent plugin with LiteLLM Proxy to build low-latency voice AI agents.

The LiveKit Agents framework provides tools for building real-time voice and video AI applications. By routing through LiteLLM Proxy, you get unified access to multiple realtime voice providers, cost tracking, rate limiting, and more.

## Quick Start

### 1. Install Dependencies

```bash
pip install livekit-agents[xai]
```

### 2. Start LiteLLM Proxy

Create a config file with your xAI realtime model:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: grok-voice-agent
    litellm_params:
      model: xai/grok-2-vision-1212
      api_key: os.environ/XAI_API_KEY
    model_info:
      mode: realtime

litellm_settings:
  drop_params: True

general_settings:
  master_key: sk-1234  # Change this to a secure key
```

Start the proxy:

```bash
litellm --config config.yaml --port 4000
```

### 3. Configure LiveKit xAI Plugin

Point LiveKit's xAI plugin to your LiteLLM proxy:

```python
from livekit.plugins import xai

# Configure xAI to use LiteLLM proxy
model = xai.realtime.RealtimeModel(
    voice="ara",                      # Voice option
    api_key="sk-1234",               # Your LiteLLM proxy master key
    base_url="http://localhost:4000", # LiteLLM proxy URL
)
```

## Complete Example

Here's a complete working example:

<Tabs>
<TabItem value="python" label="Python Client">

```python
#!/usr/bin/env python3
"""
Simple xAI realtime voice agent through LiteLLM proxy.
"""
import asyncio
import json
import websockets

PROXY_URL = "ws://localhost:4000/v1/realtime"
API_KEY = "sk-1234"
MODEL = "grok-voice-agent"

async def run_voice_agent():
    """Connect to xAI realtime API through LiteLLM proxy"""
    url = f"{PROXY_URL}?model={MODEL}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    async with websockets.connect(url, extra_headers=headers) as ws:
        # Wait for initial connection event
        initial = json.loads(await ws.recv())
        print(f"✅ Connected: {initial['type']}")
        
        # Send user message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Hello! Tell me a joke."
                }]
            }
        }))
        
        # Request response
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        }))
        
        # Collect response
        transcript = []
        async for message in ws:
            event = json.loads(message)
            
            # Capture text response
            if event['type'] == 'response.output_audio_transcript.delta':
                transcript.append(event['delta'])
                print(event['delta'], end='', flush=True)
            
            # Done when response completes
            elif event['type'] == 'response.done':
                break
        
        print(f"\n\n✅ Full response: {''.join(transcript)}")

if __name__ == "__main__":
    asyncio.run(run_voice_agent())
```

</TabItem>

<TabItem value="livekit" label="LiveKit Agent">

```python
from livekit.agents import Agent, AgentSession, WorkerOptions, cli
from livekit.plugins import xai

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant.",
            llm=xai.realtime.RealtimeModel(
                voice="ara",
                api_key="sk-1234",
                base_url="http://localhost:4000",
            ),
        )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_factory=VoiceAgent,
        )
    )
```

</TabItem>
</Tabs>

## Running the Example

1. **Start LiteLLM Proxy** (if not already running):
   ```bash
   litellm --config config.yaml --port 4000
   ```

2. **Run the example**:
   ```bash
   python your_script.py
   ```

## Expected Output

```
✅ Connected: conversation.created
Hello! Here's a joke for you: Why don't scientists trust atoms? 
Because they make up everything!

✅ Full response: Hello! Here's a joke for you: Why don't scientists trust atoms? Because they make up everything!
```


## Complete Working Example

**[LiveKit Agent SDK Cookbook](https://github.com/BerriAI/litellm/tree/main/cookbook/livekit_agent_sdk)**


## Learn More

- [xAI Realtime API](/docs/providers/xai_realtime)
- [LiveKit xAI Plugin](https://docs.livekit.io/agents/models/realtime/plugins/xai/)
- [LiteLLM Realtime API](/docs/realtime)
