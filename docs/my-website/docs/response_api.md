import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /responses [Beta]


LiteLLM provides a BETA endpoint in the spec of [OpenAI's `/responses` API](https://platform.openai.com/docs/api-reference/responses)

Requests to /chat/completions may be bridged here automatically when the provider lacks support for that endpoint. The model’s default `mode` determines how bridging works.(see `model_prices_and_context_window`) 

| Feature | Supported | Notes |
|---------|-----------|--------|
| Cost Tracking | ✅ | Works with all supported models |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Streaming | ✅ | |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Supported operations | Create a response, Get a response, Delete a response | |
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

#### GET a Response
```python showLineNumbers title="Get Response by ID"
import litellm

# First, create a response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

# Get the response ID
response_id = response.id

# Retrieve the response by ID
retrieved_response = litellm.get_responses(
    response_id=response_id
)

print(retrieved_response)

# For async usage
# retrieved_response = await litellm.aget_responses(response_id=response_id)
```

#### CANCEL a Response
You can cancel an in-progress response (if supported by the provider):

```python showLineNumbers title="Cancel Response by ID"
import litellm

# First, create a response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

# Get the response ID
response_id = response.id

# Cancel the response by ID
cancel_response = litellm.cancel_responses(
    response_id=response_id
)

print(cancel_response)

# For async usage
# cancel_response = await litellm.acancel_responses(response_id=response_id)
```


**REST API:**
```bash
curl -X POST http://localhost:4000/v1/responses/response_id/cancel \
    -H "Authorization: Bearer sk-1234"
```

This will attempt to cancel the in-progress response with the given ID.
**Note:** Not all providers support response cancellation. If unsupported, an error will be raised.

#### DELETE a Response
```python showLineNumbers title="Delete Response by ID"
import litellm

# First, create a response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

# Get the response ID
response_id = response.id

# Delete the response by ID
delete_response = litellm.delete_responses(
    response_id=response_id
)

print(delete_response)

# For async usage
# delete_response = await litellm.adelete_responses(response_id=response_id)
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

#### GET a Response
```python showLineNumbers title="Get Response by ID with OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# First, create a response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

# Get the response ID
response_id = response.id

# Retrieve the response by ID
retrieved_response = client.responses.retrieve(response_id)

print(retrieved_response)
```

#### DELETE a Response
```python showLineNumbers title="Delete Response by ID with OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# First, create a response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

# Get the response ID
response_id = response.id

# Delete the response by ID
delete_response = client.responses.delete(response_id)

print(delete_response)
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

## Load Balancing with Session Continuity.

When using the Responses API with multiple deployments of the same model (e.g., multiple Azure OpenAI endpoints), LiteLLM provides session continuity. This ensures that follow-up requests using a `previous_response_id` are routed to the same deployment that generated the original response.


#### Example Usage

<Tabs>
<TabItem value="python-sdk" label="Python SDK">

```python showLineNumbers title="Python SDK with Session Continuity"
import litellm

# Set up router with multiple deployments of the same model
router = litellm.Router(
    model_list=[
        {
            "model_name": "azure-gpt4-turbo",
            "litellm_params": {
                "model": "azure/gpt-4-turbo",
                "api_key": "your-api-key-1",
                "api_version": "2024-06-01",
                "api_base": "https://endpoint1.openai.azure.com",
            },
        },
        {
            "model_name": "azure-gpt4-turbo",
            "litellm_params": {
                "model": "azure/gpt-4-turbo",
                "api_key": "your-api-key-2",
                "api_version": "2024-06-01",
                "api_base": "https://endpoint2.openai.azure.com",
            },
        },
    ],
    optional_pre_call_checks=["responses_api_deployment_check"],
)

# Initial request
response = await router.aresponses(
    model="azure-gpt4-turbo",
    input="Hello, who are you?",
    truncation="auto",
)

# Store the response ID
response_id = response.id

# Follow-up request - will be automatically routed to the same deployment
follow_up = await router.aresponses(
    model="azure-gpt4-turbo",
    input="Tell me more about yourself",
    truncation="auto",
    previous_response_id=response_id  # This ensures routing to the same deployment
)
```

</TabItem>
<TabItem value="proxy-server" label="Proxy Server">

#### 1. Setup session continuity on proxy config.yaml

To enable session continuity for Responses API in your LiteLLM proxy, set `optional_pre_call_checks: ["responses_api_deployment_check"]` in your proxy config.yaml.

```yaml showLineNumbers title="config.yaml with Session Continuity"
model_list:
  - model_name: azure-gpt4-turbo
    litellm_params:
      model: azure/gpt-4-turbo
      api_key: your-api-key-1
      api_version: 2024-06-01
      api_base: https://endpoint1.openai.azure.com
  - model_name: azure-gpt4-turbo
    litellm_params:
      model: azure/gpt-4-turbo
      api_key: your-api-key-2
      api_version: 2024-06-01
      api_base: https://endpoint2.openai.azure.com

router_settings:
  optional_pre_call_checks: ["responses_api_deployment_check"]
```

#### 2. Use the OpenAI Python SDK to make requests to LiteLLM Proxy

```python showLineNumbers title="OpenAI Client with Proxy Server"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-api-key"
)

# Initial request
response = client.responses.create(
    model="azure-gpt4-turbo",
    input="Hello, who are you?"
)

response_id = response.id

# Follow-up request - will be automatically routed to the same deployment
follow_up = client.responses.create(
    model="azure-gpt4-turbo",
    input="Tell me more about yourself",
    previous_response_id=response_id  # This ensures routing to the same deployment
)
```

</TabItem>
</Tabs>

## Calling non-Responses API endpoints (`/responses` to `/chat/completions` Bridge)

LiteLLM allows you to call non-Responses API models via a bridge to LiteLLM's `/chat/completions` endpoint. This is useful for calling Anthropic, Gemini and even non-Responses API OpenAI models.


#### Python SDK Usage

```python showLineNumbers title="SDK Usage"
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

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Example Configuration"
model_list:
- model_name: anthropic-model
  litellm_params:
    model: anthropic/claude-3-5-sonnet-20240620
    api_key: os.environ/ANTHROPIC_API_KEY
```

**Start Proxy:**

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Make Request:**

```bash showLineNumbers title="non-Responses API Model Request"
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "anthropic-model",
    "input": "who is Michael Jordan"
  }'
```







## Session Management

LiteLLM Proxy supports session management for all supported models. This allows you to store and fetch conversation history (state) in LiteLLM Proxy. 

#### Usage

1. Enable storing request / response content in the database

Set `store_prompts_in_cold_storage: true` in your proxy config.yaml. When this is enabled, LiteLLM will store the request and response content in the s3 bucket you specify.

```yaml showLineNumbers title="config.yaml with Session Continuity"
litellm_settings:
  callbacks: ["s3_v2"]
  cold_storage_custom_logger: s3_v2
  s3_callback_params: # learn more https://docs.litellm.ai/docs/proxy/logging#s3-buckets
    s3_bucket_name: litellm-logs   # AWS Bucket Name for S3
    s3_region_name: us-west-2      

general_settings:
  store_prompts_in_cold_storage: true
  store_prompts_in_spend_logs: true
```

2. Make request 1 with no `previous_response_id` (new session)

Start a new conversation by making a request without specifying a previous response ID.

<Tabs>
<TabItem value="curl" label="Curl">

```curl
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet-latest",
    "input": "who is Michael Jordan"
  }'
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python
from openai import OpenAI

# Initialize the client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

# Make initial request to start a new conversation
response = client.responses.create(
    model="anthropic/claude-3-5-sonnet-latest",
    input="who is Michael Jordan"
)

print(response.id)  # Store this ID for future requests in same session
print(response.output[0].content[0].text)
```

</TabItem>
</Tabs>

Response:

```json
{
  "id":"resp_123abc",
  "model":"claude-3-5-sonnet-20241022",
  "output":[{
    "type":"message",
    "content":[{
      "type":"output_text",
      "text":"Michael Jordan is widely considered one of the greatest basketball players of all time. He played for the Chicago Bulls (1984-1993, 1995-1998) and Washington Wizards (2001-2003), winning 6 NBA Championships with the Bulls."
    }]
  }]
}
```

3. Make request 2 with `previous_response_id` (same session)

Continue the conversation by referencing the previous response ID to maintain conversation context.

<Tabs>
<TabItem value="curl" label="Curl">

```curl
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet-latest",
    "input": "can you tell me more about him",
    "previous_response_id": "resp_123abc"
  }'
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python
from openai import OpenAI

# Initialize the client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

# Make follow-up request in the same conversation session
follow_up_response = client.responses.create(
    model="anthropic/claude-3-5-sonnet-latest",
    input="can you tell me more about him",
    previous_response_id="resp_123abc"  # ID from the previous response
)

print(follow_up_response.output[0].content[0].text)
```

</TabItem>
</Tabs>

Response:

```json
{
  "id":"resp_456def",
  "model":"claude-3-5-sonnet-20241022",
  "output":[{
    "type":"message",
    "content":[{
      "type":"output_text",
      "text":"Michael Jordan was born February 17, 1963. He attended University of North Carolina before being drafted 3rd overall by the Bulls in 1984. Beyond basketball, he built the Air Jordan brand with Nike and later became owner of the Charlotte Hornets."
    }]
  }]
}
```

4. Make request 3 with no `previous_response_id` (new session)

Start a brand new conversation without referencing previous context to demonstrate how context is not maintained between sessions.

<Tabs>
<TabItem value="curl" label="Curl">

```curl
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet-latest",
    "input": "can you tell me more about him"
  }'
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python
from openai import OpenAI

# Initialize the client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

# Make a new request without previous context
new_session_response = client.responses.create(
    model="anthropic/claude-3-5-sonnet-latest",
    input="can you tell me more about him"
    # No previous_response_id means this starts a new conversation
)

print(new_session_response.output[0].content[0].text)
```

</TabItem>
</Tabs>

Response:

```json
{
  "id":"resp_789ghi",
  "model":"claude-3-5-sonnet-20241022",
  "output":[{
    "type":"message",
    "content":[{
      "type":"output_text",
      "text":"I don't see who you're referring to in our conversation. Could you let me know which person you'd like to learn more about?"
    }]
  }]
}
```













