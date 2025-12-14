import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SAP Generative AI Hub

LiteLLM supports SAP Generative AI Hub's Orchestration Service.

| Property | Details |
|-------|-------|
| Description | SAP's Generative AI Hub provides access to foundation models through the AI Core orchestration service. |
| Provider Route on LiteLLM | `sap/` |
| Supported Endpoints | `/chat/completions` |
| API Reference | [SAP AI Core Documentation](https://help.sap.com/docs/sap-ai-core) |

## Authentication

SAP Generative AI Hub uses service key authentication. You can provide credentials via:

1. **Environment variable** - Set `AICORE_SERVICE_KEY` with your service key JSON
2. **Direct parameter** - Pass `api_key` with the service key JSON string

```python showLineNumbers title="Environment Variable"
import os
os.environ["AICORE_SERVICE_KEY"] = '{"clientid": "...", "clientsecret": "...", ...}'
```

## Usage - LiteLLM Python SDK

```python showLineNumbers title="SAP Chat Completion"
from litellm import completion
import os

os.environ["AICORE_SERVICE_KEY"] = '{"clientid": "...", "clientsecret": "...", ...}'

response = completion(
    model="sap/gpt-4",
    messages=[{"role": "user", "content": "Hello from LiteLLM"}]
)
print(response)
```

```python showLineNumbers title="SAP Chat Completion - Streaming"
from litellm import completion
import os

os.environ["AICORE_SERVICE_KEY"] = '{"clientid": "...", "clientsecret": "...", ...}'

response = completion(
    model="sap/gpt-4",
    messages=[{"role": "user", "content": "Hello from LiteLLM"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Usage - LiteLLM Proxy

Add to your LiteLLM Proxy config:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: sap-gpt4
    litellm_params:
      model: sap/gpt-4
      api_key: os.environ/AICORE_SERVICE_KEY
```

Start the proxy:

```bash showLineNumbers title="Start Proxy"
litellm --config config.yaml
```

<Tabs>
<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Test Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "sap-gpt4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="OpenAI SDK"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-proxy-api-key"
)

response = client.chat.completions.create(
    model="sap-gpt4",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Supported Parameters

| Parameter | Description |
|-----------|-------------|
| `temperature` | Controls randomness |
| `max_tokens` | Maximum tokens in response |
| `top_p` | Nucleus sampling |
| `tools` | Function calling tools |
| `tool_choice` | Tool selection behavior |
| `response_format` | Output format (json_object, json_schema) |
| `stream` | Enable streaming |

