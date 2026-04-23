import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Clarifai
Anthropic, OpenAI, Qwen, xAI, Gemini and most of Open soured LLMs are Supported on Clarifai.

| Property | Details |
|-------|-------|
| Description | Clarifai is a powerful AI platform that provides access to a wide range of LLMs through a unified API. LiteLLM enables seamless integration with Clarifai's models using an OpenAI-compatible interface. |
| Provider Doc | [Clarifai ↗](https://docs.clarifai.com/) |
|OpenAI compatible Endpoint for Provider | `https://api.clarifai.com/v2/ext/openai/v1` |
| Supported Endpoints | `/chat/completions` |

## Pre-Requisites

```bash
pip install litellm
```

## Required Environment Variables
To obtain your Clarifai Personal access token follow this [link](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/).

```python
os.environ["CLARIFAI_PAT"] = "CLARIFAI_API_KEY"  # CLARIFAI_PAT
```

## Usage

```python
import os
from litellm import completion

os.environ["CLARIFAI_API_KEY"] = ""

response = completion(
  model="clarifai/openai.chat-completion.gpt-oss-20b",
  messages=[{ "content": "Tell me a joke about physics?","role": "user"}]
)
```
## Streaming Support

LiteLLM supports streaming responses with Clarifai models:

```python
import litellm

for chunk in litellm.completion(
    model="clarifai/openai.chat-completion.gpt-oss-20b",
    api_key="CLARIFAI_API_KEY",
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
  }
}]

response = litellm.completion(
    model="clarifai/openai.chat-completion.gpt-oss-20b",
    api_key="CLARIFAI_API_KEY",
    messages=[{"role": "user", "content": "What is the weather in Paris today?"}],
    tools=tools,
)

print(response.choices[0].message.tool_calls)
```

## Clarifai models
liteLLM supports all models on [Clarifai community](https://clarifai.com/explore/models?filterData=%5B%7B%22field%22%3A%22use_cases%22%2C%22value%22%3A%5B%22llm%22%5D%7D%5D&page=1&perPage=24)

### 🧠 OpenAI Models
- [gpt-oss-20b](https://clarifai.com/openai/chat-completion/models/gpt-oss-20b)
- [gpt-oss-120b](https://clarifai.com/openai/chat-completion/models/gpt-oss-120b)
- [gpt-5-nano](https://clarifai.com/openai/chat-completion/models/gpt-5-nano)
- [gpt-5-mini](https://clarifai.com/openai/chat-completion/models/gpt-5-mini)
- [gpt-5](https://clarifai.com/openai/chat-completion/models/gpt-5)
- [gpt-4o](https://clarifai.com/openai/chat-completion/models/gpt-4o)
- [o3](https://clarifai.com/openai/chat-completion/models/o3)
- Many more...


### 🤖 Anthropic Models
- [claude-sonnet-4](https://clarifai.com/anthropic/completion/models/claude-sonnet-4)
- [claude-opus-4](https://clarifai.com/anthropic/completion/models/claude-opus-4)
- [claude-3_5-haiku](https://clarifai.com/anthropic/completion/models/claude-3_5-haiku)
- [claude-3_7-sonnet](https://clarifai.com/anthropic/completion/models/claude-3_7-sonnet)
- Many more...


### 🪄 xAI Models
- [grok-3](https://clarifai.com/xai/chat-completion/models/grok-3)
- [grok-2-vision-1212](https://clarifai.com/xai/chat-completion/models/grok-2-vision-1212)
- [grok-2-1212](https://clarifai.com/xai/chat-completion/models/grok-2-1212)
- [grok-code-fast-1](https://clarifai.com/xai/chat-completion/models/grok-code-fast-1)
- [grok-2-image-1212](https://clarifai.com/xai/image-generation/models/grok-2-image-1212)
- Many more...


### 🔷 Google Gemini Models
- [gemini-2_5-pro](https://clarifai.com/gcp/generate/models/gemini-2_5-pro)
- [gemini-2_5-flash-lite](https://clarifai.com/gcp/generate/models/gemini-2_5-flash-lite)
- [gemini-2_0-flash](https://clarifai.com/gcp/generate/models/gemini-2_0-flash)
- [gemini-2_0-flash-lite](https://clarifai.com/gcp/generate/models/gemini-2_0-flash-lite)
- Many more...


### 🧩 Qwen Models
- [Qwen3-30B-A3B-Instruct-2507](https://clarifai.com/qwen/qwenLM/models/Qwen3-30B-A3B-Instruct-2507)
- [Qwen3-30B-A3B-Thinking-2507](https://clarifai.com/qwen/qwenLM/models/Qwen3-30B-A3B-Thinking-2507)
- [Qwen3-14B](https://clarifai.com/qwen/qwenLM/models/Qwen3-14B)
- [QwQ-32B-AWQ](https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ)
- [Qwen2_5-VL-7B-Instruct](https://clarifai.com/qwen/qwen-VL/models/Qwen2_5-VL-7B-Instruct)
- [Qwen3-Coder-30B-A3B-Instruct](https://clarifai.com/qwen/qwenCoder/models/Qwen3-Coder-30B-A3B-Instruct)
- Many more...


### 💡 MiniCPM (OpenBMB) Models
- [MiniCPM-o-2_6-language](https://clarifai.com/openbmb/miniCPM/models/MiniCPM-o-2_6-language)
- [MiniCPM3-4B](https://clarifai.com/openbmb/miniCPM/models/MiniCPM3-4B)
- [MiniCPM4-8B](https://clarifai.com/openbmb/miniCPM/models/MiniCPM4-8B)
- Many more...


### 🧬 Microsoft Phi Models
- [Phi-4-reasoning-plus](https://clarifai.com/microsoft/text-generation/models/Phi-4-reasoning-plus)
- [phi-4](https://clarifai.com/microsoft/text-generation/models/phi-4)
- Many more...


### 🦙 Meta Llama Models
- [Llama-3_2-3B-Instruct](https://clarifai.com/meta/Llama-3/models/Llama-3_2-3B-Instruct)
- Many more...


### 🔍 DeepSeek Models
- [DeepSeek-R1-0528-Qwen3-8B](https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1-0528-Qwen3-8B)
- Many more...

## Usage with LiteLLM Proxy

Here's how to call Clarifai with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export CLARIFAI_PAT="CLARIFAI_API_KEY"
```

### 2. Start the proxy

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: clarifai-model
    litellm_params:
      model: clarifai/openai.chat-completion.gpt-oss-20b
      api_key: os.environ/CLARIFAI_PAT
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

## Important Notes

- Always prefix Clarifai model IDs with `clarifai/` when specifying the model name
- Use your Clarifai Personal Access Token (PAT) as the API key
- Usage is tracked and billed through Clarifai
- API rate limits are subject to your Clarifai account settings
- Most OpenAI parameters are supported, but some advanced features may vary by model


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