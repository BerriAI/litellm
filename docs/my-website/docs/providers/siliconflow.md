import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SiliconFlow

| Property | Details |
|-------|-------|
| Description | SiliconFlow is an AI cloud platform that helps developers easily deploy AI models through a simple API, backed by affordable and reliable GPU cloud infrastructure. LiteLLM supports all models from [SiliconFlow](https://www.siliconflow.com/models?utm_source=github&utm_medium=referral&utm_term=github_readme&utm_content=github_litellm) |
| Provider Route on LiteLLM | `siliconflow/` |
| Provider Doc | [SiliconFlow Docs ↗](https://docs.siliconflow.com/) |
| API Endpoint for Provider | https://api.siliconflow.com/v1 |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions` |

<br />

## API Keys

Get your API key [here](https://cloud.siliconflow.com/me/account/ak).

```python
import os
os.environ["SILICONFLOW_API_KEY"] = "your-api-key"
```

## Supported OpenAI Params
- max_tokens
- stream
- stream_options
- n
- frequency_penalty
- presence_penalty
- repetition_penalty
- stop
- temperature
- top_p
- top_k
- min_p
- tools
- response_format

## Sample Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion
os.environ["SILICONFLOW_API_KEY"] = "your-api-key"

response = completion(
    model="siliconflow/deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "List 5 popular cookie recipes."}]
)

content = response.get('choices', [{}])[0].get('message', {}).get('content')
print(content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: deepseek-ai/DeepSeek-V3
    litellm_params:
      model: siliconflow/deepseek-ai/DeepSeek-V3
      api_key: os.environ/SILICONFLOW_API_KEY
```

2. Start Proxy

```
$ litellm --config /path/to/config.yaml
```

3. Make Request!


```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
  "model": "deepseek-ai/DeepSeek-V3",
  "messages": [
      {"role": "user", "content": "List 5 popular cookie recipes."}
  ]
}
'
```

</TabItem>
</Tabs>


## Tool Calling

```python
from litellm import completion
import os
# set env
os.environ["SILICONFLOW_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

response = completion(
    model="siliconflow/deepseek-ai/DeepSeek-V3",
    messages=messages,
    tools=tools,
)
# Add any assertions, here to check response args
print(response)
assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
assert isinstance(
    response.choices[0].message.tool_calls[0].function.arguments, str
)

```

## JSON Mode

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import json
import os

os.environ['SILICONFLOW_API_KEY'] = "your-api-key"

messages = [
    {
        "role": "user",
        "content": "List 5 popular cookie recipes in a JSON array."
    }
]

completion = completion(
    model="siliconflow/deepseek-ai/DeepSeek-V3",
    messages=messages,
    response_format={"type": "json_object"} # 👈 KEY CHANGE
)

print(json.loads(completion.choices[0].message.content))
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: deepseek-ai/DeepSeek-V3
    litellm_params:
      model: siliconflow/deepseek-ai/DeepSeek-V3
      api_key: os.environ/SILICONFLOW_API_KEY
```

2. Start Proxy

```
$ litellm --config /path/to/config.yaml
```

3. Make Request!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
  "model": "deepseek-ai/DeepSeek-V3",
  "messages": [
      {"role": "user", "content": "List 5 popular cookie recipes in a JSON array."}
  ],
  "response_format": {"type": "json_object"}
}
'
```

</TabItem>
</Tabs>

## Chat Models

🚨 LiteLLM supports ALL SiliconFlow models, send `model=siliconflow/<your-siliconflow-model>` to send it to SiliconFlow. See all SiliconFlow models [here](https://www.siliconflow.com/models?utm_source=github&utm_medium=referral&utm_term=github_readme&utm_content=github_litellm).

| Model Name                | Function Call                                       |
|---------------------------|-----------------------------------------------------|
| deepseek-ai/DeepSeek-V3 | `completion('siliconflow/deepseek-ai/DeepSeek-V3', messages)` |
| deepseek-ai/DeepSeek-R1 | `completion('siliconflow/deepseek-ai/DeepSeek-R1', messages)` |
| zai-org/GLM-4.6 | `completion('siliconflow/zai-org/GLM-4.6', messages)` |
| moonshotai/Kimi-K2-Instruct | `completion('siliconflow/moonshotai/Kimi-K2-Instruct', messages)` |
