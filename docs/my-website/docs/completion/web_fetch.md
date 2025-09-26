import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Web Fetch

Web fetch allows models to autonomously browse and retrieve content from web pages. This enables AI models to access real-time information from the internet and incorporate web content into their responses.

**Supported Providers:**
- Anthropic API (`anthropic/`)

**Supported Tool Types:**
- `web_fetch_20250910` - Web content retrieval tool with usage limits

LiteLLM will standardize the web fetch tools across all supported providers.

## Quick Start

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Web fetch tool
tools = [
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
    }
]

messages = [
    {
        "role": "user", 
        "content": "Please analyze the content at https://example.com/article and summarize the main points"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy Server">

1. Define web fetch models on config.yaml

```yaml
model_list:
  - model_name: claude-3-5-sonnet-latest # Anthropic claude-3-5-sonnet-latest
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-bedrock         # Bedrock Anthropic model
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
    model_info:
      supports_web_fetch: True        # set supports_web_fetch to True so /model/info returns this attribute as True
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK

```python
import os 
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # your litellm proxy api key
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-latest",
    messages=[
        {
            "role": "user", 
            "content": "Please fetch and analyze the content from https://news.ycombinator.com and tell me about the top stories"
        }
    ],
    tools=[
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 5,
        }
    ]
)

print(response)
```

</TabItem>
</Tabs>

