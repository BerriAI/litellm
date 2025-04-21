import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /responses [Beta]

LiteLLM provides a BETA endpoint in the spec of [OpenAI's `/responses` API](https://platform.openai.com/docs/api-reference/responses)

| Feature | Supported | Notes |
|---------|-----------|--------|
| Cost Tracking | ✅ | Works with all supported models |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Streaming | ✅ | |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Supported LiteLLM Versions | 1.63.8+ | |
| Supported LLM providers | **All LiteLLM supported providers** | `openai`, `anthropic`, `bedrock`, `vertex_ai`, `gemini`, `azure`, `azure_ai` etc. |

## Usage

### LiteLLM Python SDK

<Tabs>
<TabItem value="openai" label="OpenAI">

#### Non-streaming
```python showLineNumbers title="OpenAI Non-streaming Response"
import litellm

# Non-streaming response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

print(response)
```

#### Streaming
```python showLineNumbers title="OpenAI Streaming Response"
import litellm

# Streaming response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

#### Non-streaming
```python showLineNumbers title="Anthropic Non-streaming Response"
import litellm
import os

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Non-streaming response
response = litellm.responses(
    model="anthropic/claude-3-5-sonnet-20240620",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

print(response)
```

#### Streaming
```python showLineNumbers title="Anthropic Streaming Response"
import litellm
import os

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Streaming response
response = litellm.responses(
    model="anthropic/claude-3-5-sonnet-20240620",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="vertex" label="Vertex AI">

#### Non-streaming
```python showLineNumbers title="Vertex AI Non-streaming Response"
import litellm
import os

# Set credentials - Vertex AI uses application default credentials
# Run 'gcloud auth application-default login' to authenticate
os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# Non-streaming response
response = litellm.responses(
    model="vertex_ai/gemini-1.5-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

print(response)
```

#### Streaming
```python showLineNumbers title="Vertex AI Streaming Response"
import litellm
import os

# Set credentials - Vertex AI uses application default credentials
# Run 'gcloud auth application-default login' to authenticate
os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# Streaming response
response = litellm.responses(
    model="vertex_ai/gemini-1.5-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="bedrock" label="AWS Bedrock">

#### Non-streaming
```python showLineNumbers title="AWS Bedrock Non-streaming Response"
import litellm
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-access-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"  # or your AWS region

# Non-streaming response
response = litellm.responses(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

print(response)
```

#### Streaming
```python showLineNumbers title="AWS Bedrock Streaming Response"
import litellm
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-access-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"  # or your AWS region

# Streaming response
response = litellm.responses(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="gemini" label="Google AI Studio">

#### Non-streaming
```python showLineNumbers title="Google AI Studio Non-streaming Response"
import litellm
import os

# Set API key for Google AI Studio
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

# Non-streaming response
response = litellm.responses(
    model="gemini/gemini-1.5-flash",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

print(response)
```

#### Streaming
```python showLineNumbers title="Google AI Studio Streaming Response"
import litellm
import os

# Set API key for Google AI Studio
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

# Streaming response
response = litellm.responses(
    model="gemini/gemini-1.5-flash",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>
</Tabs>

### LiteLLM Proxy with OpenAI SDK

First, set up and start your LiteLLM proxy server.

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai" label="OpenAI">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/o1-pro
    litellm_params:
      model: openai/o1-pro
      api_key: os.environ/OPENAI_API_KEY
```

#### Non-streaming
```python showLineNumbers title="OpenAI Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers title="OpenAI Proxy Streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="Anthropic Proxy Configuration"
model_list:
  - model_name: anthropic/claude-3-5-sonnet-20240620
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY
```

#### Non-streaming
```python showLineNumbers title="Anthropic Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="anthropic/claude-3-5-sonnet-20240620",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers title="Anthropic Proxy Streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="anthropic/claude-3-5-sonnet-20240620",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="vertex" label="Vertex AI">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="Vertex AI Proxy Configuration"
model_list:
  - model_name: vertex_ai/gemini-1.5-pro
    litellm_params:
      model: vertex_ai/gemini-1.5-pro
      vertex_project: your-gcp-project-id
      vertex_location: us-central1
```

#### Non-streaming
```python showLineNumbers title="Vertex AI Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="vertex_ai/gemini-1.5-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers title="Vertex AI Proxy Streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="vertex_ai/gemini-1.5-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="bedrock" label="AWS Bedrock">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="AWS Bedrock Proxy Configuration"
model_list:
  - model_name: bedrock/anthropic.claude-3-sonnet-20240229-v1:0
    litellm_params:
      model: bedrock/anthropic.claude-3-sonnet-20240229-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
```

#### Non-streaming
```python showLineNumbers title="AWS Bedrock Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers title="AWS Bedrock Proxy Streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>

<TabItem value="gemini" label="Google AI Studio">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="Google AI Studio Proxy Configuration"
model_list:
  - model_name: gemini/gemini-1.5-flash
    litellm_params:
      model: gemini/gemini-1.5-flash
      api_key: os.environ/GEMINI_API_KEY
```

#### Non-streaming
```python showLineNumbers title="Google AI Studio Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="gemini/gemini-1.5-flash",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers title="Google AI Studio Proxy Streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="gemini/gemini-1.5-flash",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

</TabItem>
</Tabs>

## Supported Responses API Parameters

| Provider | Supported Parameters |
|----------|---------------------|
| `openai` | [All Responses API parameters are supported](https://github.com/BerriAI/litellm/blob/7c3df984da8e4dff9201e4c5353fdc7a2b441831/litellm/llms/openai/responses/transformation.py#L23) |
| `azure` | [All Responses API parameters are supported](https://github.com/BerriAI/litellm/blob/7c3df984da8e4dff9201e4c5353fdc7a2b441831/litellm/llms/openai/responses/transformation.py#L23) |
| `anthropic` | [See supported parameters here](https://github.com/BerriAI/litellm/blob/f39d9178868662746f159d5ef642c7f34f9bfe5f/litellm/responses/litellm_completion_transformation/transformation.py#L57) |
| `bedrock` | [See supported parameters here](https://github.com/BerriAI/litellm/blob/f39d9178868662746f159d5ef642c7f34f9bfe5f/litellm/responses/litellm_completion_transformation/transformation.py#L57) |
| `gemini` | [See supported parameters here](https://github.com/BerriAI/litellm/blob/f39d9178868662746f159d5ef642c7f34f9bfe5f/litellm/responses/litellm_completion_transformation/transformation.py#L57) |
| `vertex_ai` | [See supported parameters here](https://github.com/BerriAI/litellm/blob/f39d9178868662746f159d5ef642c7f34f9bfe5f/litellm/responses/litellm_completion_transformation/transformation.py#L57) |
| `azure_ai` | [See supported parameters here](https://github.com/BerriAI/litellm/blob/f39d9178868662746f159d5ef642c7f34f9bfe5f/litellm/responses/litellm_completion_transformation/transformation.py#L57) |
| All other llm api providers | [See supported parameters here](https://github.com/BerriAI/litellm/blob/f39d9178868662746f159d5ef642c7f34f9bfe5f/litellm/responses/litellm_completion_transformation/transformation.py#L57) |

