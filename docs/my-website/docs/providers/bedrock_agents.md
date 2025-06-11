import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock Agents

Call Bedrock Agents in the OpenAI Request/Response format.


| Property | Details |
|----------|---------|
| Description | Amazon Bedrock Agents use the reasoning of foundation models (FMs), APIs, and data to break down user requests, gather relevant information, and efficiently complete tasks. |
| Provider Route on LiteLLM | `bedrock/agent/{AGENT_ID}/{ALIAS_ID}` |
| Provider Doc | [AWS Bedrock Agents ↗](https://aws.amazon.com/bedrock/agents/) |

## Quick Start

### Model Format to LiteLLM

To call a bedrock agent through LiteLLM, you need to use the following model format to call the agent.

Here the `model=bedrock/agent/` tells LiteLLM to call the bedrock `InvokeAgent` API.

```shell showLineNumbers title="Model Format to LiteLLM"
bedrock/agent/{AGENT_ID}/{ALIAS_ID}
```

**Example:**
- `bedrock/agent/L1RT58GYRW/MFPSBCXYTW`
- `bedrock/agent/ABCD1234/LIVE`

You can find these IDs in your AWS Bedrock console under Agents.


### LiteLLM Python SDK

```python showLineNumbers title="Basic Agent Completion"
import litellm

# Make a completion request to your Bedrock Agent
response = litellm.completion(
    model="bedrock/agent/L1RT58GYRW/MFPSBCXYTW",  # agent/{AGENT_ID}/{ALIAS_ID}
    messages=[
        {
            "role": "user", 
            "content": "Hi, I need help with analyzing our Q3 sales data and generating a summary report"
        }
    ],
)

print(response.choices[0].message.content)
print(f"Response cost: ${response._hidden_params['response_cost']}")
```

```python showLineNumbers title="Streaming Agent Responses"
import litellm

# Stream responses from your Bedrock Agent
response = litellm.completion(
    model="bedrock/agent/L1RT58GYRW/MFPSBCXYTW",
    messages=[
        {
            "role": "user",
            "content": "Can you help me plan a marketing campaign and provide step-by-step execution details?"
        }
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```


### LiteLLM Proxy

#### 1. Configure your model in config.yaml

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="LiteLLM Proxy Configuration"
model_list:
  - model_name: bedrock-agent-1
    litellm_params:
      model: bedrock/agent/L1RT58GYRW/MFPSBCXYTW
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2

  - model_name: bedrock-agent-2  
    litellm_params:
      model: bedrock/agent/AGENT456/ALIAS789
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

</TabItem>
</Tabs>

#### 2. Start the LiteLLM Proxy

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

#### 3. Make requests to your Bedrock Agents

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Basic Agent Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "bedrock-agent-1",
    "messages": [
      {
        "role": "user", 
        "content": "Analyze our customer data and suggest retention strategies"
      }
    ]
  }'
```

```bash showLineNumbers title="Streaming Agent Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "bedrock-agent-2",
    "messages": [
      {
        "role": "user",
        "content": "Create a comprehensive social media strategy for our new product"
      }
    ],
    "stream": true
  }'
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Using OpenAI SDK with LiteLLM Proxy"
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Make a completion request to your agent
response = client.chat.completions.create(
    model="bedrock-agent-1",
    messages=[
      {
        "role": "user",
        "content": "Help me prepare for the quarterly business review meeting"
      }
    ]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Streaming with OpenAI SDK"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000", 
    api_key="your-litellm-api-key"
)

# Stream agent responses
stream = client.chat.completions.create(
    model="bedrock-agent-2",
    messages=[
      {
        "role": "user",
        "content": "Walk me through launching a new feature beta program"
      }
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>
</Tabs>

## Further Reading

- [AWS Bedrock Agents Documentation](https://aws.amazon.com/bedrock/agents/)
- [LiteLLM Authentication to Bedrock](https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication)
