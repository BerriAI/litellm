import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SAP Generative AI Hub

LiteLLM supports SAP Generative AI Hub's Orchestration Service.

| Property | Details                                                                                                                                                |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Description | SAP's Generative AI Hub provides access to OpenAI, Anthropic, Gemini, Mistral, NVIDIA, Amazon, and SAP LLMs through the AI Core orchestration service. |
| Provider Route on LiteLLM | `sap/`                                                                                                                                                 |
| Supported Endpoints | `/chat/completions`, `/embeddings`                                                                                                                                  |
| API Reference | [SAP AI Core Documentation](https://help.sap.com/docs/sap-ai-core)                                                                                     |

## Authentication

SAP Generative AI Hub uses a service key for authentication, which can be provided in two ways.

> **Note on Precedence:** If both methods are configured, LiteLLM will prioritize the individual environment variables (`AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, etc.).

<Tabs>
<TabItem value="individual-vars" label="Recommended: Individual Variables">

Set the following variables in your environment or a `.env` file. This method is recommended for its clarity and ease of use in most environments.

```plaintext title=".env"
AICORE_AUTH_URL="https://<your-instance>.authentication.sap.hana.ondemand.com/oauth/token"
AICORE_CLIENT_ID="your-client-id"
AICORE_CLIENT_SECRET="your-client-secret"
AICORE_RESOURCE_GROUP="your-resource-group"
AICORE_BASE_URL="https://api.ai.<your-region>.cfapps.sap.hana.ondemand.com/v2"
```

</TabItem>
<TabItem value="service-key-json" label="Alternative: JSON Object">

Alternatively, you can provide the entire service key as a single JSON string called `AICORE_SERVICE_KEY` environment variable. This can be useful in environments where managing a single variable is easier.

The JSON object must include `clientid`, `clientsecret`, `url`, `apiurl`, and `resourcegroup`.

</TabItem>
</Tabs>

## Usage - LiteLLM Python SDK

:::important Model Naming for Direct SDK Usage
When using the LiteLLM Python SDK directly (not through the proxy), you **must** include the `sap/` prefix in the model name. This prefix tells LiteLLM to route your request to the SAP Gen AI Hub provider.

Example: `model="sap/anthropic--claude-4.5-sonnet"`
:::

The SDK will automatically detect which authentication method you have configured.

<Tabs>
<TabItem value="sdk-individual-vars" label="With Individual Variables">

If you have set the individual environment variables (`AICORE_AUTH_URL`, etc.), you can make calls directly.

```python showLineNumbers title="SAP Chat Completion"
from litellm import completion

# Assumes AICORE_AUTH_URL, AICORE_CLIENT_ID, etc. are set in your environment
response = completion(
    model="sap/anthropic--claude-4.5-sonnet",
    messages=[{"role": "user", "content": "Hello from LiteLLM"}]
)
print(response)
```

</TabItem>
<TabItem value="sdk-service-key-json" label="With JSON Object">

If you are using the `AICORE_SERVICE_KEY` variable, the setup is the same.

```python showLineNumbers title="SAP Chat Completion"
from litellm import completion
import os
import json

# Set the AICORE_SERVICE_KEY environment variable
service_key = {
    "clientid": "...", "clientsecret": "...", "url": "...", 
    "apiurl": "...", "resourcegroup": "..."
}
os.environ["AICORE_SERVICE_KEY"] = json.dumps(service_key)

response = completion(
    model="sap/anthropic--claude-4.5-sonnet",
    messages=[{"role": "user", "content": "Hello from LiteLLM"}]
)
print(response)
```

</TabItem>
</Tabs>

## Usage - LiteLLM Proxy

:::important Model Naming for Proxy Usage
When using clients through the LiteLLM Proxy, you use the **friendly `model_name`** defined in your proxy configuration (e.g., `claude-4.5-sonnet`, `gpt-5`). The proxy internally handles routing to the correct SAP provider using the `sap/` prefix specified in `litellm_params.model`.

**Do not** include the `sap/` prefix in your client requests - the proxy configuration handles this automatically.
:::

You can configure the proxy to use either authentication method.

<Tabs>
<TabItem value="proxy-individual-vars" label="Recommended: Individual Variables">

Add the individual environment variables to your `config.yaml`.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-5
    litellm_params:
      model: sap/gpt-5
  - model_name: gemini-2.5-pro
    litellm_params:
      model: sap/gemini-2.5-pro
  - model_name: claude-4.5-sonnet
    litellm_params:
      model: sap/anthropic--claude-4.5-sonnet
  - model_name: text-embedding-3-small
    litellm_params:
      model: sap/text-embedding-3-small

general_settings:
  master_key: your-proxy-api-key

environment_variables:
  AICORE_AUTH_URL: "https://<your-instance>.authentication.sap.hana.ondemand.com/oauth/token"
  AICORE_CLIENT_ID: "your-client-id"
  AICORE_CLIENT_SECRET: "your-client-secret"
  AICORE_RESOURCE_GROUP: "your-resource-group"
  AICORE_BASE_URL: "https://api.ai.<your-region>.cfapps.sap.hana.ondemand.com/v2"
```

</TabItem>
<TabItem value="proxy-service-key-json" label="Alternative: JSON Object">

Provide the `AICORE_SERVICE_KEY` as a single string in your `config.yaml`.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-5
    litellm_params:
      model: sap/gpt-5
  - model_name: gemini-2.5-pro
    litellm_params:
      model: sap/gemini-2.5-pro
  - model_name: claude-4.5-sonnet
    litellm_params:
      model: sap/anthropic--claude-4.5-sonnet
  - model_name: text-embedding-3-small
    litellm_params:
      model: sap/text-embedding-3-small

general_settings:
  master_key: your-proxy-api-key

environment_variables:
  AICORE_SERVICE_KEY: '{"clientid": "...", "clientsecret": "...", "url": "...", "apiurl": "...", "resourcegroup": "..."}'
```

</TabItem>
</Tabs>

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
    "model": "gpt-5",
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
    model="gpt-5",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="LiteLLM SDK"
import os
import litellm
os.environ["LITELLM_PROXY_API_KEY"] = "your-proxy-api-key"
litellm.use_litellm_proxy = True  # it is important to set this parameter
response = litellm.completion(
    model="claude-4.5-sonnet",
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    api_base="http://your-proxy-api-base"
)

print(response)
```

</TabItem>
</Tabs>

## Advanced Usage

### Streaming Responses

SAP Gen AI Hub supports streaming for real-time response generation:

```python showLineNumbers title="Streaming Chat Completion"
from litellm import completion

response = completion(
    model="sap/gpt-4o",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Response Format

SAP Gen AI Hub supports structured JSON output through response formatting.

#### JSON Schema (Recommended)

Use `json_schema` for structured output with schema validation:

```python showLineNumbers title="JSON Schema Response"
from litellm import completion

response = completion(
    model="sap/gpt-4o",
    messages=[{
        "role": "user",
        "content": "Generate info about a city with name, population, and country"
    }],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "city_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "population": {"type": "number"},
                    "country": {"type": "string"}
                },
                "required": ["name", "population", "country"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

print(response.choices[0].message.content)
# Output: {"name":"Tokyo","population":37000000,"country":"Japan"}
```

#### JSON Object Format

When using `response_format: {type: "json_object"}`, SAP requires that your prompt explicitly mentions "json" to ensure intentional JSON output:

```python showLineNumbers title="JSON Object Response"
from litellm import completion

response = completion(
    model="sap/gpt-4o",
    messages=[{
        "role": "user",
        "content": "Generate a person object in JSON format with name and age fields"
    }],
    response_format={"type": "json_object"}
)

print(response.choices[0].message.content)
```

:::note SAP Platform Requirement
When using `json_object` type, SAP's orchestration service requires the word "json" to appear in your prompt. This ensures users explicitly request JSON formatting. Alternatively, use `json_schema` (recommended) which doesn't have this requirement.
:::

### Multi-turn Conversations

SAP Gen AI Hub maintains context across conversation turns:

```python showLineNumbers title="Multi-turn Conversation"
from litellm import completion

response = completion(
    model="sap/gpt-4o",
    messages=[
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"}
    ]
)

print(response.choices[0].message.content)
# Output: Your name is Alice.
```

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

## Supported Models

SAP AI Core provides access to models from multiple providers including OpenAI, Anthropic, Google Gemini, Mistral, Amazon, Meta Llama, and NVIDIA.

For the complete list of available models, refer to the [SAP AI Core Generative AI Hub documentation](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/models-and-scenarios-in-generative-ai-hub).

> **Note:** Anthropic models use the `anthropic--` prefix (double dashes), such as `sap/anthropic--claude-4.5-sonnet`. Model availability varies by SAP deployment and region.

