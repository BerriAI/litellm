# Clarifai

Anthropic, OpenAI, Mistral, Llama, xAI, Gemini and most of Open soured LLMs are Supported on Clarifai.

| Property | Details |
|-------|-------|
| Description | Clarifai is a powerful AI platform that provides access to a wide range of LLMs through a unified API. LiteLLM enables seamless integration with Clarifai's models using an OpenAI-compatible interface. |
| Provider Route on LiteLLM | `openai/` (add this prefix to the Clarifai model URL, e.g. `openai/https://clarifai.com/openai/chat-completion/models/o4-mini`) |
| Provider Doc | [Clarifai ↗](https://docs.clarifai.com/) |
| API Endpoint for Provider | `https://api.clarifai.com/v2/ext/openai/v1` |
| Supported Endpoints | `/chat/completions` |

## Benefits

- **Unified Interface:** Use the same OpenAI-compatible syntax across multiple LLM providers
- **Lightweight and Fast:** Simple, performant, and easy to configure
- **Flexible Deployment:** Integrate Clarifai's specialized models in existing or new workflows
- **Wide Model Selection:** Access to a diverse range of models from various providers

## Pre-Requisites

```bash
pip install litellm
```

## Required Environment Variables

To obtain your Clarifai Personal access token follow this [link](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/).

```python
os.environ["CLARIFAI_PAT"] = "CLARIFAI_API_KEY"  # CLARIFAI_PAT
```

## Basic Usage

```python
import os
from litellm import completion

response = completion(
    model="openai/https://clarifai.com/openai/chat-completion/models/o4-mini",
    api_base="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ["CLARIFAI_PAT"],
    messages=[{ "content": "Tell me a joke about physics?","role": "user"}]
)
```

## Usage with LiteLLM Proxy

Here's how to call Clarifai with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export CLARIFAI_PAT="your-pat"
```

### 2. Start the proxy

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: clarifai-model
    litellm_params:
      model: openai/https://clarifai.com/openai/chat-completion/models/o4-mini
      api_key: os.environ/CLARIFAI_PAT
      api_base: https://api.clarifai.com/v2/ext/openai/v1
```

```bash
litellm --config /path/to/config.yaml

# Server running on http://0.0.0.0:4000
```
</TabItem>
</Tabs>

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "clarifai-model",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="clarifai-model",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
```
</TabItem>
</Tabs>

## Streaming Support

LiteLLM supports streaming responses with Clarifai models:

```python
import litellm

for chunk in litellm.completion(
    model="openai/https://clarifai.com/openai/chat-completion/models/o4-mini",
    api_key="CLARIFAI_API_KEY",
    api_base="https://api.clarifai.com/v2/ext/openai/v1",
    messages=[
        {"role": "user", "content": "Tell me a fun fact about space."}
    ],
    stream=True,
):
    print(chunk.choices[0].delta)
```

## Tool Calling (Function Calling)

Clarifai models accessed via LiteLLM support function calling:

```python
import litellm

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Tokyo, Japan"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
    }
}]

response = litellm.completion(
    model="openai/https://clarifai.com/openai/chat-completion/models/o4-mini",
    api_key="CLARIFAI_API_KEY",
    api_base="https://api.clarifai.com/v2/ext/openai/v1",
    messages=[{"role": "user", "content": "What is the weather in Paris today?"}],
    tools=tools,
)

print(response.choices[0].message.tool_calls)
```

## Important Notes

- Always prefix Clarifai model URLs with `openai/` when specifying the model name
- You must set `api_base` to `https://api.clarifai.com/v2/ext/openai/v1`
- Use your Clarifai Personal Access Token (PAT) as the API key
- Usage is tracked and billed through Clarifai
- API rate limits are subject to your Clarifai account settings
- Most OpenAI parameters are supported, but some advanced features may vary by model

## Supported Models

Clarifai provides access to a wide range of models through their OpenAI-compatible interface. Here are some popular options:

### OpenAI Models
- [gpt-4_1](https://clarifai.com/openai/chat-completion/models/gpt-4_1)
- [o3](https://clarifai.com/openai/chat-completion/models/o3)
- [o4-mini](https://clarifai.com/openai/chat-completion/models/o4-mini)
- [gpt-4o](https://clarifai.com/openai/chat-completion/models/gpt-4o)
- Many more...

### Anthropic Models
- [claude-sonnet-4](https://clarifai.com/anthropic/completion/models/claude-sonnet-4)
- [claude-3_5-haiku](https://clarifai.com/anthropic/completion/models/claude-3_5-haiku)
- [claude-opus-4](https://clarifai.com/anthropic/completion/models/claude-opus-4)
- [claude-3_5-sonnet](https://clarifai.com/anthropic/completion/models/claude-3_5-sonnet)
- [claude-3_7-sonnet](https://clarifai.com/anthropic/completion/models/claude-3_7-sonnet)
- Many more...

### xAI Models
- [grok-3](https://clarifai.com/xai/chat-completion/models/grok-3)
- [grok-2-vision-1212](https://clarifai.com/xai/chat-completion/models/grok-2-vision-1212)
- Many more...

### Gemini Models
- [gemini-2_5-pro](https://clarifai.com/gcp/generate/models/gemini-2_5-pro)
- [gemini-2_5-flash](https://clarifai.com/gcp/generate/models/gemini-2_5-flash)
- [gemini-2_0-flash](https://clarifai.com/gcp/generate/models/gemini-2_0-flash)
- [gemini-2_0-flash-lite](https://clarifai.com/gcp/generate/models/gemini-2_0-flash-lite)
- [gemini-pro](https://clarifai.com/gcp/generate/models/gemini-pro)
- Many more...

### Open Sourced Models
- [DeepSeek-R1-0528-Qwen3-8B](https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1-0528-Qwen3-8B)
- [DeepSeek-R1-Distill-Qwen-7B](https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1](https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1)
- [DeepSeek-R1-Distill-Qwen-32B](https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1-Distill-Qwen-32B)
- [Devstral-Small-2505](https://clarifai.com/mistralai/completion/models/Devstral-Small-2505_gguf-4bit)
- [gemma-3-4b-it](https://clarifai.com/gcp/generate/models/gemma-3-4b-it)
- [gemma-3-12b-it](https://clarifai.com/gcp/generate/models/gemma-3-12b-it)
- [gemma-3-1b-it](https://clarifai.com/gcp/generate/models/gemma-3-1b-it)
- Many more...

For a complete list of available models, visit the [Clarifai Model Explorer](https://clarifai.com/explore/).

## FAQs

| Question | Answer |
|----------|---------|
| Can I use all Clarifai models with LiteLLM? | Most chat-completion models are supported. Use the Clarifai model URL as the `model`. |
| Do I need a separate Clarifai PAT? | Yes, you must use a valid Clarifai Personal Access Token. |
| Is tool calling supported? | Yes, provided the underlying Clarifai model supports function/tool calling. |
| How is billing handled? | Clarifai usage is billed independently via Clarifai. |

## Additional Resources

- [Clarifai Documentation](https://docs.clarifai.com/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Clarifai Runners Examples](https://github.com/Clarifai/runners-examples)
