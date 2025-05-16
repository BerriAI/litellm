import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nscale (EU Sovereign)

https://docs.nscale.com/docs/inference/chat

:::tip

**We support ALL Nscale models, just set `model=nscale/<any-model-on-nscale>` as a prefix when sending litellm requests**

:::

| Property | Details |
|-------|-------|
| Description | European-domiciled full-stack AI cloud platform for LLMs and image generation. |
| Provider Route on LiteLLM | `nscale/` |
| Supported Endpoints | `/chat/completions`, `/images/generations` |
| API Reference | [Nscale docs](https://docs.nscale.com/docs/getting-started/overview) |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["NSCALE_API_KEY"] = ""  # your Nscale API key
```

## Explore Available Models  

Explore our full list of text and multimodal AI models — all available at highly competitive pricing:
📚 [Full List of Models](https://docs.nscale.com/docs/inference/serverless-models/current)  


## Key Features
- **EU Sovereign**: Full data sovereignty and compliance with European regulations
- **Ultra-Low Cost (starting at $0.01 / M tokens)**: Extremely competitive pricing for both text and image generation models
- **Production Grade**: Reliable serverless deployments with full isolation
- **No Setup Required**: Instant access to compute without infrastructure management
- **Full Control**: Your data remains private and isolated

## Usage - LiteLLM Python SDK

### Text Generation

```python showLineNumbers title="Nscale Text Generation"
from litellm import completion
import os

os.environ["NSCALE_API_KEY"] = ""  # your Nscale API key
response = completion(
    model="nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[{"role": "user", "content": "What is LiteLLM?"}]
)
print(response)
```

```python showLineNumbers title="Nscale Text Generation - Streaming"
from litellm import completion
import os

os.environ["NSCALE_API_KEY"] = ""  # your Nscale API key
stream = completion(
    model="nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[{"role": "user", "content": "What is LiteLLM?"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Image Generation

```python showLineNumbers title="Nscale Image Generation"
from litellm import image_generation
import os

os.environ["NSCALE_API_KEY"] = ""  # your Nscale API key
response = image_generation(
    model="nscale/stabilityai/stable-diffusion-xl-base-1.0",
    prompt="A beautiful sunset over mountains",
    n=1,
    size="1024x1024"
)
print(response)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct
    litellm_params:
      model: nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct
      api_key: os.environ/NSCALE_API_KEY
  - model_name: nscale/meta-llama/Llama-3.3-70B-Instruct
    litellm_params:
      model: nscale/meta-llama/Llama-3.3-70B-Instruct
      api_key: os.environ/NSCALE_API_KEY
  - model_name: nscale/stabilityai/stable-diffusion-xl-base-1.0
    litellm_params:
      model: nscale/stabilityai/stable-diffusion-xl-base-1.0
      api_key: os.environ/NSCALE_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Nscale via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[{"role": "user", "content": "What is LiteLLM?"}]
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Nscale via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[{"role": "user", "content": "What is LiteLLM?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Nscale via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{"role": "user", "content": "What is LiteLLM?"}]
  }'
```

</TabItem>
</Tabs>

## Getting Started
1. Create an account at [console.nscale.com](https://console.nscale.com)
2. Claim free credit
3. Create an API key in settings
4. Start making API calls using LiteLLM

## Additional Resources
- [Nscale Documentation](https://docs.nscale.com/docs/getting-started/overview)
- [Blog: Sovereign Serverless](https://www.nscale.com/blog/sovereign-serverless-how-we-designed-full-isolation-without-sacrificing-performance) 
