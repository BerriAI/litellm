import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# IO Intelligence

[IO.NET](https://io.net) is a decentralized GPU ecosystem that provides AI infrastructure and computing resources at up to 70% cost savings compared to traditional cloud providers. [IO Intelligence](https://docs.io.net/reference/get-started-with-io-intelligence-api) offers access to open-source machine learning models deployed on IO.NET's hardware infrastructure with instant access to 30,000+ GPUs.

:::tip

**We support ALL IO Intelligence models, just set `model=io_intelligence/<any-model-on-io-intelligence>` as a prefix when sending litellm requests.**

:::

## API Key

```python
# env variable
os.environ['IO_INTELLIGENCE_API_KEY'] = "your-api-key"
```

## Sample Usage

```python
from litellm import completion
import os

os.environ['IO_INTELLIGENCE_API_KEY'] = "your-api-key"

response = completion(
    model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hi, I am doing a project using IO Intelligence.",
        }
    ],
    max_tokens=50,
    temperature=0.7,
)
print(response)
```

## Sample Usage - Streaming

```python
from litellm import completion
import os

os.environ['IO_INTELLIGENCE_API_KEY'] = "your-api-key"

response = completion(
    model="io_intelligence/meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What do you know about IO.NET?",
        }
    ],
    stream=True,
    max_tokens=100,
    temperature=0.7,
)

for chunk in response:
    print(chunk)
```

## Available Models

IO Intelligence provides access to various open-source models including:

- **meta-llama/Llama-3.3-70B-Instruct**
- **mistralai/Mistral-Large-Instruct-2411**  
- **Qwen/Qwen3-235B-A22B-Thinking**
- **deepseek-ai/DeepSeek-R1-0528**
- get more details here: [IO Intelligence Models](https://docs.io.net/reference/models)
    - https://api.intelligence.io.solutions/api/v1/models

## Usage Limits

- **Daily token limits**: 1,000,000 tokens for chat completions per account per model
- **Rate limits**: 
  - Chat: 3 requests/second
  - Embeddings: 10 requests/second
- **API interactions**: 500,000 tokens daily

## Usage with LiteLLM Proxy Server

Here's how to call IO Intelligence models with the LiteLLM Proxy Server

1. Modify the config.yaml 

```yaml
model_list:
  - model_name: llama-3.3-70b
    litellm_params:
      model: io_intelligence/meta-llama/Llama-3.3-70B-Instruct
      api_key: os.environ/IO_INTELLIGENCE_API_KEY
  - model_name: mistral-large
    litellm_params:
      model: io_intelligence/mistralai/Mistral-Large-Instruct-2411
      api_key: os.environ/IO_INTELLIGENCE_API_KEY
```

2. Start the proxy 

```bash
$ litellm --config /path/to/config.yaml
```

3. Send Request to LiteLLM Proxy Server

<Tabs>

<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to LiteLLM
response = client.chat.completions.create(
    model="llama-3.3-70b", # model alias from config
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ # pass params not supported by OpenAI
        "max_completion_tokens": 50,
        "temperature": 0.7
    }
)

print(response)
```
</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
     "model": "llama-3.3-70b",
     "messages": [
       {
         "role": "user",
         "content": "this is a test request, write a short poem"
       }
     ],
     "max_completion_tokens": 50,
     "temperature": 0.7
}'
```
</TabItem>

</Tabs>

## Embeddings

```python
from litellm import embedding
import os

os.environ['IO_INTELLIGENCE_API_KEY'] = "your-api-key"

response = embedding(
    model="io_intelligence/text-embedding-ada-002",
    input=["good morning from litellm", "this is another item"],
)
```

## Environment Variables

```python
import os

# Required
os.environ["IO_INTELLIGENCE_API_KEY"] = "your-api-key"

# Optional - if not provided, defaults to https://api.intelligence.io.solutions/api/v1
os.environ["IO_INTELLIGENCE_API_BASE"] = "https://api.intelligence.io.solutions/api/v1"
```

## Supported Parameters

All OpenAI Chat Completion parameters are supported including:

- `temperature`
- `max_tokens` / `max_completion_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `stop`
- `stream`
- `response_format`
- `tools` and `tool_choice`
- `function_calling`

## About IO.NET

IO.NET is a decentralized GPU ecosystem designed to provide AI infrastructure at significantly reduced costs:

- **Instant Access**: 30,000+ GPUs available without waitlists
- **Cost Savings**: Up to 70% cheaper than AWS
- **Flexible Deployment**: Support for containers, VMs, and Ray clusters  
- **Pay-as-you-go**: Only pay for what you use
- **Compliance**: AICPA SOC certified and GDPR compliant
- **GPU Types**: GeForce RTX 4090, H100 PCIe, H100SXM5, H200
- **Pricing**: Starting as low as $0.25/hour

The platform democratizes GPU access for AI startups, enterprise teams, and individual developers while allowing GPU providers to monetize idle computing resources.

## Getting Started

1. **Sign up**: Create an account at [IO.NET](https://io.net)
2. **Get API Key**: Generate your API key from the dashboard
3. **Set Environment Variable**: `export IO_INTELLIGENCE_API_KEY="your-key"`
4. **Start Building**: Use with LiteLLM as shown in the examples above

For more detailed documentation, visit the [IO Intelligence API Reference](https://docs.io.net/reference/get-started-with-io-intelligence-api).