import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI - Response API

## Usage

### LiteLLM Python SDK


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


### LiteLLM Proxy with OpenAI SDK

1. Set up config.yaml

```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/o1-pro
    litellm_params:
      model: openai/o1-pro
      api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Use OpenAI SDK with LiteLLM Proxy

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


## Supported Responses API Parameters

| Provider | Supported Parameters |
|----------|---------------------|
| `openai` | [All Responses API parameters are supported](https://github.com/BerriAI/litellm/blob/7c3df984da8e4dff9201e4c5353fdc7a2b441831/litellm/llms/openai/responses/transformation.py#L23) |

## Computer Use 

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import litellm

# Non-streaming response
response = litellm.responses(
    model="computer-use-preview",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser" # other possible values: "mac", "windows", "ubuntu"
    }],    
    input=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Check the latest OpenAI news on bing.com."
            }
            # Optional: include a screenshot of the initial state of the environment
            # {
            #     type: "input_image",
            #     image_url: f"data:image/png;base64,{screenshot_base64}"
            # }
          ]
        }
    ],
    reasoning={
        "summary": "concise",
    },
    truncation="auto"
)

print(response.output)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Set up config.yaml

```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/o1-pro
    litellm_params:
      model: openai/o1-pro
      api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```python showLineNumbers title="OpenAI Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="computer-use-preview",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser" # other possible values: "mac", "windows", "ubuntu"
    }],    
    input=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Check the latest OpenAI news on bing.com."
            }
            # Optional: include a screenshot of the initial state of the environment
            # {
            #     type: "input_image",
            #     image_url: f"data:image/png;base64,{screenshot_base64}"
            # }
          ]
        }
    ],
    reasoning={
        "summary": "concise",
    },
    truncation="auto"
)

print(response)
```


</TabItem>
</Tabs>


## MCP Tools 

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python showLineNumbers title="MCP Tools with LiteLLM SDK"
import litellm
from typing import Optional

# Configure MCP Tools
MCP_TOOLS = [
    {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "allowed_tools": ["ask_question"]
    }
]

# Step 1: Make initial request - OpenAI will use MCP LIST and return MCP calls for approval
response = litellm.responses(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input="What transport protocols does the 2025-03-26 version of the MCP spec support?"
)

# Get the MCP approval ID
mcp_approval_id = None
for output in response.output:
    if output.type == "mcp_approval_request":
        mcp_approval_id = output.id
        break

# Step 2: Send followup with approval for the MCP call
response_with_mcp_call = litellm.responses(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input=[
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": mcp_approval_id
        }
    ],
    previous_response_id=response.id,
)

print(response_with_mcp_call)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Set up config.yaml

```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
      api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```python showLineNumbers title="MCP Tools with OpenAI SDK via LiteLLM Proxy"
from openai import OpenAI
from typing import Optional

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Configure MCP Tools
MCP_TOOLS = [
    {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "allowed_tools": ["ask_question"]
    }
]

# Step 1: Make initial request - OpenAI will use MCP LIST and return MCP calls for approval
response = client.responses.create(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input="What transport protocols does the 2025-03-26 version of the MCP spec support?"
)

# Get the MCP approval ID
mcp_approval_id = None
for output in response.output:
    if output.type == "mcp_approval_request":
        mcp_approval_id = output.id
        break

# Step 2: Send followup with approval for the MCP call
response_with_mcp_call = client.responses.create(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input=[
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": mcp_approval_id
        }
    ],
    previous_response_id=response.id,
)

print(response_with_mcp_call)
```

</TabItem>
</Tabs>


