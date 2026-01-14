# Azure Model Router

Azure Model Router is a feature in Azure AI Foundry that automatically routes your requests to the best available model based on your requirements. This allows you to use a single endpoint that intelligently selects the optimal model for each request.

## Key Features

- **Automatic Model Selection**: Azure Model Router dynamically selects the best model for your request
- **Cost Tracking**: LiteLLM automatically tracks costs based on the actual model used (e.g., `gpt-4.1-nano`), not the router endpoint
- **Streaming Support**: Full support for streaming responses with accurate cost calculation

## LiteLLM Python SDK

### Basic Usage

```python
import litellm
import os

response = litellm.completion(
    model="azure_ai/azure-model-router",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="https://your-endpoint.cognitiveservices.azure.com/openai/v1/",
    api_key=os.getenv("AZURE_MODEL_ROUTER_API_KEY"),
)

print(response)
```

### Streaming with Usage Tracking

```python
import litellm
import os

response = await litellm.acompletion(
    model="azure_ai/azure-model-router",
    messages=[{"role": "user", "content": "hi"}],
    api_base="https://your-endpoint.cognitiveservices.azure.com/openai/v1/",
    api_key=os.getenv("AZURE_MODEL_ROUTER_API_KEY"),
    stream=True,
    stream_options={"include_usage": True},
)

async for chunk in response:
    print(chunk)
```

## LiteLLM Proxy (AI Gateway)

### config.yaml

```yaml
model_list:
  - model_name: azure-model-router
    litellm_params:
      model: azure_ai/azure-model-router
      api_base: https://your-endpoint.cognitiveservices.azure.com/openai/v1/
      api_key: os.environ/AZURE_MODEL_ROUTER_API_KEY
```

### Start Proxy

```bash
litellm --config config.yaml
```

### Test Request

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "azure-model-router",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Add Azure Model Router via LiteLLM UI

This walkthrough shows how to add an Azure Model Router endpoint to LiteLLM using the Admin Dashboard.

### Step 1: Navigate to Models Page

Go to the Models page in your LiteLLM Dashboard.

![Navigate to Models](./img/azure_model_router_01.jpeg)

### Step 2: Select Provider

Click the "Provider" dropdown field.

![Click Provider](./img/azure_model_router_02.jpeg)

### Step 3: Choose Azure AI Foundry

Select "Azure AI Foundry (Studio)" from the provider list.

![Select Azure AI Foundry](./img/azure_model_router_03.jpeg)

### Step 4: Configure Model Name

Click on the model name field to configure your model.

![Click Model Field](./img/azure_model_router_04.jpeg)

### Step 5: Select Custom Model Name

Choose "Custom Model Name (Enter below)" to enter a custom model identifier.

![Select Custom Model](./img/azure_model_router_05.jpeg)

### Step 6: Enter LiteLLM Model Name

Click on "LiteLLM Model Name(s)" to specify the model name that will be used in API calls.

![LiteLLM Model Name](./img/azure_model_router_06.jpeg)

### Step 7: Enter Custom Model Name

Click the "Enter custom model name" field.

![Enter Custom Name Field](./img/azure_model_router_07.jpeg)

### Step 8: Type Model Prefix

Type `azure_ai/` as the prefix for your model name.

![Type azure_ai prefix](./img/azure_model_router_08.jpeg)

### Step 9: Get Model Name from Azure Portal

Switch to your Azure AI Foundry portal and locate your model router deployment name.

![Azure Portal Model Name](./img/azure_model_router_09.jpeg)

### Step 10: Copy Model Name

Copy the model router name (e.g., `azure-model-router`) from the Azure portal.

![Copy Model Name](./img/azure_model_router_10.jpeg)

### Step 11: Paste Model Name

Paste the model name into the LiteLLM Dashboard field, resulting in `azure_ai/azure-model-router`.

![Paste Model Name](./img/azure_model_router_11.jpeg)

### Step 12: Get API Base URL

Go back to the Azure portal and copy the endpoint URL for your model router.

![Copy API Base](./img/azure_model_router_12.jpeg)

### Step 13: Enter API Base

Click the "API Base" field in the LiteLLM Dashboard.

![Click API Base Field](./img/azure_model_router_13.jpeg)

### Step 14: Paste API Base URL

Paste the endpoint URL from Azure.

![Paste API Base](./img/azure_model_router_14.jpeg)

### Step 15: Get API Key

Copy your API key from the Azure portal.

![Copy API Key](./img/azure_model_router_15.jpeg)

### Step 16: Enter API Key

Click the "Azure API Key" field and paste your API key.

![Enter API Key](./img/azure_model_router_16.jpeg)

### Step 17: Test Connection

Click "Test Connect" to verify your configuration works correctly.

![Test Connection](./img/azure_model_router_17.jpeg)

### Step 18: Close Test Dialog

After successful test, click "Close" to dismiss the test dialog.

![Close Dialog](./img/azure_model_router_18.jpeg)

### Step 19: Add Model

Click "Add Model" to save your Azure Model Router configuration.

![Add Model](./img/azure_model_router_19.jpeg)

### Step 20: Test in Playground

Navigate to the "Playground" to test your newly added model.

![Go to Playground](./img/azure_model_router_20.jpeg)

### Step 21: Select Model

Type "azure" to filter and select your Azure Model Router model.

![Select Model](./img/azure_model_router_21.jpeg)

### Step 22: Send Test Message

Type a test message in the chat field and send it.

![Send Message](./img/azure_model_router_22.jpeg)

### Step 23: View Logs

Click "Logs" to see the request details and cost tracking.

![View Logs](./img/azure_model_router_23.jpeg)

### Step 24: Verify Cost Tracking

You can see the cost is tracked correctly based on the actual model used by Azure Model Router.

![Verify Cost](./img/azure_model_router_24.jpeg)

## Cost Tracking

LiteLLM automatically handles cost tracking for Azure Model Router by:

1. **Detecting the actual model**: When Azure Model Router routes your request to a specific model (e.g., `gpt-4.1-nano-2025-04-14`), LiteLLM extracts this from the response
2. **Calculating accurate costs**: Costs are calculated based on the actual model used, not the router endpoint name
3. **Streaming support**: Cost tracking works correctly for both streaming and non-streaming requests

### Example Response with Cost

```python
import litellm

response = litellm.completion(
    model="azure_ai/azure-model-router",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="https://your-endpoint.cognitiveservices.azure.com/openai/v1/",
    api_key="your-api-key",
)

# The response will show the actual model used
print(f"Model used: {response.model}")  # e.g., "gpt-4.1-nano-2025-04-14"

# Get cost
from litellm import completion_cost
cost = completion_cost(completion_response=response)
print(f"Cost: ${cost}")
```

## Supported Parameters

Azure Model Router supports all standard chat completion parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Must be `azure_ai/azure-model-router` or your custom model name |
| `messages` | array | Array of message objects |
| `stream` | boolean | Enable streaming responses |
| `stream_options` | object | Options for streaming (e.g., `{"include_usage": true}`) |
| `temperature` | float | Sampling temperature |
| `max_tokens` | integer | Maximum tokens to generate |
| `top_p` | float | Nucleus sampling parameter |

## Troubleshooting

### Cost showing as None

If cost tracking shows `None`, ensure:
1. The actual model returned by Azure Model Router is in LiteLLM's pricing database
2. You're using the latest version of LiteLLM

### Streaming cost not tracked

For streaming requests, LiteLLM extracts the model from the response chunks. This is handled automatically - no additional configuration needed.

