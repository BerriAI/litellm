import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Responses API

| Property | Details |
|-------|-------|
| Description | Azure OpenAI Responses API |
| `custom_llm_provider` on LiteLLM | `azure/` |
| Supported Operations | `/v1/responses`|
| Azure OpenAI Responses API | [Azure OpenAI Responses API ↗](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses?tabs=python-secure) |
| Cost Tracking, Logging Support | ✅ LiteLLM will log, track cost for Responses API Requests |
| Supported OpenAI Params | ✅ All OpenAI params are supported, [See here](https://github.com/BerriAI/litellm/blob/0717369ae6969882d149933da48eeb8ab0e691bd/litellm/llms/openai/responses/transformation.py#L23) |

## Usage

## Create a model response

<Tabs>
<TabItem value="litellm-sdk" label="LiteLLM SDK">

#### Non-streaming

```python showLineNumbers title="Azure Responses API"
import litellm

# Non-streaming response
response = litellm.responses(
    model="azure/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100,
    api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
    api_base="https://litellm8397336933.openai.azure.com/",
    api_version="2023-03-15-preview",
)

print(response)
```

#### Streaming
```python showLineNumbers title="Azure Responses API"
import litellm

# Streaming response
response = litellm.responses(
    model="azure/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True,
    api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
    api_base="https://litellm8397336933.openai.azure.com/",
    api_version="2023-03-15-preview",
)

for event in response:
    print(event)
```

</TabItem>
<TabItem value="proxy" label="OpenAI SDK with LiteLLM Proxy">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="Azure Responses API"
model_list:
  - model_name: o1-pro
    litellm_params:
      model: azure/o1-pro
      api_key: os.environ/AZURE_RESPONSES_OPENAI_API_KEY
      api_base: https://litellm8397336933.openai.azure.com/
      api_version: 2023-03-15-preview
```

Start your LiteLLM proxy:
```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Then use the OpenAI SDK pointed to your proxy:

#### Non-streaming
```python showLineNumbers
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>
</Tabs>

