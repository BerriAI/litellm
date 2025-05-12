import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nscale (EU Sovereign)

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

## Supported Models

### Chat Models

| Model Name | Description | Input Cost | Output Cost |
|------------|-------------|------------|-------------|
| nscale/meta-llama/Llama-4-Scout-17B-16E-Instruct | 17B parameter model | $0.09/M tokens | $0.29/M tokens |
| nscale/Qwen/Qwen2.5-Coder-3B-Instruct | 3B parameter coding model | $0.01/M tokens | $0.03/M tokens |
| nscale/Qwen/Qwen2.5-Coder-7B-Instruct | 7B parameter coding model | $0.01/M tokens | $0.03/M tokens |
| nscale/Qwen/Qwen2.5-Coder-32B-Instruct | 32B parameter coding model | $0.06/M tokens | $0.20/M tokens |
| nscale/Qwen/QwQ-32B | 32B parameter model | $0.18/M tokens | $0.20/M tokens |
| nscale/deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 70B parameter distilled model | $0.375/M tokens | $0.375/M tokens |
| nscale/deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B parameter distilled model | $0.025/M tokens | $0.025/M tokens |
| nscale/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B parameter distilled model | $0.09/M tokens | $0.09/M tokens |
| nscale/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B parameter distilled model | $0.20/M tokens | $0.20/M tokens |
| nscale/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 14B parameter distilled model | $0.07/M tokens | $0.07/M tokens |
| nscale/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 32B parameter distilled model | $0.15/M tokens | $0.15/M tokens |
| nscale/mistralai/mixtral-8x22b-instruct-v0.1 | Mixtral 8x22B model | $0.60/M tokens | $0.60/M tokens |
| nscale/meta-llama/Llama-3.1-8B-Instruct | 8B parameter model | $0.03/M tokens | $0.03/M tokens |
| nscale/meta-llama/Llama-3.3-70B-Instruct | 70B parameter model | $0.20/M tokens | $0.20/M tokens |

### Image Generation Models

| Model Name | Description | Cost per Pixel |
|------------|-------------|----------------|
| nscale/black-forest-labs/FLUX.1-schnell | Fast image generation model | $0.0000000013 |
| nscale/stabilityai/stable-diffusion-xl-base-1.0 | SDXL base model | $0.000000003 |

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
2. Add credit to your account (minimum $5)
3. Create an API key in settings
4. Start making API calls using LiteLLM

## Additional Resources
- [Nscale Documentation](https://docs.nscale.com/docs/getting-started/overview)
- [Blog: Sovereign Serverless](https://www.nscale.com/blog/sovereign-serverless-how-we-designed-full-isolation-without-sacrificing-performance) 