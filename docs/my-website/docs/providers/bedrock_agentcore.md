import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock AgentCore

Call Bedrock AgentCore in the OpenAI Request/Response format.

| Property | Details |
|----------|---------|
| Description | Amazon Bedrock AgentCore provides direct access to hosted agent runtimes for executing agentic workflows with foundation models. |
| Provider Route on LiteLLM | `bedrock/agentcore/{AGENT_RUNTIME_ARN}` |
| Provider Doc | [AWS Bedrock AgentCore ↗](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agentcore_InvokeAgentRuntime.html) |

:::info

This documentation is for **AgentCore Agents** (agent runtimes). If you want to use AgentCore MCP servers, add them as you would any other MCP server. See the [MCP documentation](https://docs.litellm.ai/docs/mcp) for details.

:::

## Quick Start

### Model Format to LiteLLM

To call a bedrock agent runtime through LiteLLM, use the following model format.

Here the `model=bedrock/agentcore/` tells LiteLLM to call the bedrock `InvokeAgentRuntime` API.

```shell showLineNumbers title="Model Format to LiteLLM"
bedrock/agentcore/{AGENT_RUNTIME_ARN}
```

**Example:**
- `bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-runtime`

You can find the Agent Runtime ARN in your AWS Bedrock console under AgentCore.

### LiteLLM Python SDK

```python showLineNumbers title="Basic AgentCore Completion"
import litellm

# Make a completion request to your AgentCore runtime
response = litellm.completion(
    model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-runtime",
    messages=[
        {
            "role": "user", 
            "content": "Explain machine learning in simple terms"
        }
    ],
)

print(response.choices[0].message.content)
print(f"Usage: {response.usage}")
```

```python showLineNumbers title="Streaming AgentCore Responses"
import litellm

# Stream responses from your AgentCore runtime
response = litellm.completion(
    model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-runtime",
    messages=[
        {
            "role": "user",
            "content": "What are the key principles of software architecture?"
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
  - model_name: agentcore-runtime-1
    litellm_params:
      model: bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-runtime
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2

  - model_name: agentcore-runtime-2
    litellm_params:
      model: bedrock/agentcore/arn:aws:bedrock-agentcore:us-east-1:987654321098:runtime/production-runtime
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

#### 3. Make requests to your AgentCore runtimes

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Basic AgentCore Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "agentcore-runtime-1",
    "messages": [
      {
        "role": "user", 
        "content": "Summarize the main benefits of cloud computing"
      }
    ]
  }'
```

```bash showLineNumbers title="Streaming AgentCore Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "agentcore-runtime-2",
    "messages": [
      {
        "role": "user",
        "content": "Explain the differences between SQL and NoSQL databases"
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

# Make a completion request to your AgentCore runtime
response = client.chat.completions.create(
    model="agentcore-runtime-1",
    messages=[
      {
        "role": "user",
        "content": "What are best practices for API design?"
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

# Stream AgentCore responses
stream = client.chat.completions.create(
    model="agentcore-runtime-2",
    messages=[
      {
        "role": "user",
        "content": "Describe the microservices architecture pattern"
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

## Provider-specific Parameters

AgentCore supports additional parameters that can be passed to customize the runtime invocation.

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers title="Using AgentCore-specific parameters"
from litellm import completion

response = litellm.completion(
    model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-runtime",
    messages=[
        {
            "role": "user",
            "content": "Analyze this data and provide insights",
        }
    ],
    qualifier="production",  # PROVIDER-SPECIFIC: Runtime qualifier/version
    runtimeSessionId="session-abc-123",  # PROVIDER-SPECIFIC: Custom session ID
)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```yaml showLineNumbers title="LiteLLM Proxy Configuration with Parameters"
model_list:
  - model_name: agentcore-runtime-prod
    litellm_params:
      model: bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-runtime
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
      qualifier: production
```

</TabItem>
</Tabs>

### Available Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `qualifier` | string | Optional runtime qualifier/version to invoke a specific version of the agent runtime |
| `runtimeSessionId` | string | Optional custom session ID (must be 33+ characters). If not provided, LiteLLM generates one automatically |

## Further Reading

- [AWS Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agentcore_InvokeAgentRuntime.html)
- [LiteLLM Authentication to Bedrock](https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication)

