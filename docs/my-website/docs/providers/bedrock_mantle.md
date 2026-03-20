import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Amazon Bedrock Mantle

[Amazon Bedrock Mantle](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html) is Amazon Bedrock's distributed inference engine (Project Mantle) that exposes an **OpenAI-compatible API** for Bedrock-hosted models.

Use this provider to call Bedrock Mantle models with accurate **AWS Bedrock pricing** instead of OpenAI pricing.

:::tip

**We support ALL Bedrock Mantle models, just set `model=bedrock_mantle/<model-id>` as a prefix when sending litellm requests**

:::

## API Key

```python
# env variable
os.environ['BEDROCK_MANTLE_API_KEY'] = "your-aws-bedrock-api-key"

# optional: override region (defaults to us-east-1)
os.environ['BEDROCK_MANTLE_REGION'] = "us-east-1"  # or use AWS_REGION
```

## Supported Models

| Model | Context Window | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|---------------|----------------------|------------------------|
| `openai.gpt-oss-120b` | 131K | $0.15 | $0.60 |
| `openai.gpt-oss-20b` | 131K | $0.075 | $0.30 |
| `openai.gpt-oss-safeguard-120b` | 131K | $0.15 | $0.60 |
| `openai.gpt-oss-safeguard-20b` | 131K | $0.075 | $0.30 |

## Sample Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['BEDROCK_MANTLE_API_KEY'] = "your-bedrock-api-key"

response = completion(
    model="bedrock_mantle/openai.gpt-oss-120b",
    messages=[{"role": "user", "content": "hello from litellm"}],
)
print(response)
```

</TabItem>
<TabItem value="streaming" label="Streaming">

```python
from litellm import completion
import os

os.environ['BEDROCK_MANTLE_API_KEY'] = "your-bedrock-api-key"

response = completion(
    model="bedrock_mantle/openai.gpt-oss-120b",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

</TabItem>
<TabItem value="async" label="Async">

```python
import asyncio
from litellm import acompletion
import os

os.environ['BEDROCK_MANTLE_API_KEY'] = "your-bedrock-api-key"

async def main():
    response = await acompletion(
        model="bedrock_mantle/openai.gpt-oss-120b",
        messages=[{"role": "user", "content": "hello from litellm"}],
    )
    print(response)

asyncio.run(main())
```

</TabItem>
</Tabs>

## Region Configuration

The API base URL is `https://bedrock-mantle.{region}.api.aws/v1`. Region is resolved in this order:

1. `BEDROCK_MANTLE_REGION` env var
2. `AWS_REGION` env var
3. Default: `us-east-1`

**Supported regions:** `us-east-1`, `us-east-2`, `us-west-2`, `eu-west-1`, `eu-west-2`, `eu-central-1`, `eu-south-1`, `eu-north-1`, `ap-northeast-1`, `ap-south-1`, `ap-southeast-3`, `sa-east-1`

```python
import os
os.environ['BEDROCK_MANTLE_REGION'] = "eu-west-1"

# or pass api_base directly
response = completion(
    model="bedrock_mantle/openai.gpt-oss-120b",
    messages=[{"role": "user", "content": "hello"}],
    api_base="https://bedrock-mantle.eu-west-1.api.aws/v1",
)
```

## Usage with LiteLLM Proxy

### 1. Set Bedrock Mantle models on config.yaml

```yaml
model_list:
  - model_name: gpt-oss-120b
    litellm_params:
      model: bedrock_mantle/openai.gpt-oss-120b
      api_key: os.environ/BEDROCK_MANTLE_API_KEY
      # optional region override:
      api_base: "https://bedrock-mantle.us-east-1.api.aws/v1"

  - model_name: gpt-oss-20b
    litellm_params:
      model: bedrock_mantle/openai.gpt-oss-20b
      api_key: os.environ/BEDROCK_MANTLE_API_KEY
```

### 2. Start the proxy

```shell
litellm --config /path/to/config.yaml
```

### 3. Send a request

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000",
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "hello from litellm"}],
)
print(response)
```
