# LiveKit Voice Agent with LiteLLM Gateway

Simple example showing how to use LiveKit's xAI realtime plugin with LiteLLM as a proxy. This lets you switch between xAI, OpenAI, and Azure realtime APIs without changing your code.

## Quick Start

### 1. Install dependencies

```bash
pip install livekit-agents[xai] websockets
```

### 2. Start LiteLLM proxy

```bash
# With xAI
export XAI_API_KEY="your-xai-key"
litellm --config config.yaml --port 4000
```

### 3. Run the voice agent

```bash
python main.py
```

Type your message and get a voice response from Grok!

## Configuration

Set these environment variables if needed:

```bash
export LITELLM_PROXY_URL="http://localhost:4000"
export LITELLM_API_KEY="sk-1234"
export LITELLM_MODEL="grok-voice-agent"
```

Or use the defaults - connects to `http://localhost:4000` by default.

## Example Config File

Create a `config.yaml` with your realtime models:

```yaml
model_list:
  - model_name: grok-voice-agent
    litellm_params:
      model: xai/grok-2-vision-1212
      api_key: os.environ/XAI_API_KEY
    model_info:
      mode: realtime

  - model_name: openai-voice-agent
    litellm_params:
      model: gpt-4o-realtime-preview
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime

general_settings:
  master_key: sk-1234
```

Then start: `litellm --config config.yaml --port 4000`

## How It Works

LiveKit's xAI plugin connects through LiteLLM proxy by setting `base_url`:

```python
from livekit.plugins import xai

model = xai.realtime.RealtimeModel(
    voice="ara",
    api_key="sk-1234",              # LiteLLM proxy key
    base_url="http://localhost:4000", # Point to LiteLLM
)
```

## Switching Providers

Just change the model in your config - no code changes needed:

**xAI Grok:**
```yaml
model: xai/grok-2-vision-1212
```

**OpenAI:**
```yaml
model: gpt-4o-realtime-preview
```

**Azure OpenAI:**
```yaml
model: azure/gpt-4o-realtime-preview
api_base: https://your-endpoint.openai.azure.com/
```

## Why Use LiteLLM?

- ✅ **Switch providers** without changing agent code
- ✅ **Cost tracking** across all voice sessions
- ✅ **Rate limiting** and budgets
- ✅ **Load balancing** across multiple API keys
- ✅ **Fallbacks** to backup models

## Learn More

- [LiveKit xAI Realtime Tutorial](/docs/tutorials/livekit_xai_realtime)
- [xAI Realtime Docs](/docs/providers/xai_realtime)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiteLLM Realtime API](/docs/realtime)
