# Bedrock (boto3) SDK

Pass-through endpoints for Bedrock - call provider-specific endpoint, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ | For `/invoke` and `/converse` endpoints |
| Load Balancing | ✅ | You can load balance `/invoke`, `/converse` routes across multiple deployments| Logging | ✅ | works across all integrations |
| End-user Tracking | ❌ | [Tell us if you need this](https://github.com/BerriAI/litellm/issues/new) |
| Streaming | ✅ | |

Just replace `https://bedrock-runtime.{aws_region_name}.amazonaws.com` with `LITELLM_PROXY_BASE_URL/bedrock` 🚀

## Overview

LiteLLM supports two ways to call Bedrock endpoints:

### 1. **Using config.yaml** (Recommended for model endpoints)

Define your Bedrock models in `config.yaml` and reference them by name. The proxy handles authentication and routing.

**Use for**: `/converse`, `/converse-stream`, `/invoke`, `/invoke-with-response-stream`

```yaml showLineNumbers
model_list:
  - model_name: my-bedrock-model
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_region_name: us-west-2
      custom_llm_provider: bedrock
```

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/my-bedrock-model/converse' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}'
```

### 2. **Direct passthrough** (For non-model endpoints)

Set AWS credentials via environment variables and call Bedrock endpoints directly.

**Use for**: Guardrails, Knowledge Bases, Agents, and other non-model endpoints

```bash showLineNumbers
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_REGION_NAME="us-west-2"
```

```bash showLineNumbers
curl "http://0.0.0.0:4000/bedrock/guardrail/my-guardrail-id/version/1/apply" \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{"contents": [{"text": {"text": "Hello"}}], "source": "INPUT"}'
```

Supports **ALL** Bedrock Endpoints (including streaming).

[**See All Bedrock Endpoints**](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)

## Quick Start

Let's call the Bedrock [`/converse` endpoint](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)

1. Create a `config.yaml` file with your Bedrock model

```yaml showLineNumbers
model_list:
  - model_name: my-bedrock-model
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_region_name: us-west-2
      custom_llm_provider: bedrock
```

Set your AWS credentials:

```bash showLineNumbers
export AWS_ACCESS_KEY_ID=""  # Access key
export AWS_SECRET_ACCESS_KEY="" # Secret access key
```

2. Start LiteLLM Proxy 

```bash showLineNumbers
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Bedrock converse endpoint using the model name from config:

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/my-bedrock-model/converse' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 100
    }
}'
```

## Setup with config.yaml

Use config.yaml to define Bedrock models and use them via passthrough endpoints.

### 1. Define models in config.yaml

```yaml showLineNumbers
model_list:
  - model_name: my-claude-model
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_region_name: us-west-2
      custom_llm_provider: bedrock
  
  - model_name: my-cohere-model
    litellm_params:
      model: bedrock/cohere.command-r-v1:0
      aws_region_name: us-east-1
      custom_llm_provider: bedrock
```

### 2. Start proxy with config

```bash showLineNumbers
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Call Bedrock Converse endpoint

Use the `model_name` from config in the URL path:

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/my-claude-model/converse' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}]
        }
    ],
    "inferenceConfig": {
        "temperature": 0.5,
        "maxTokens": 100
    }
}'
```

### 4. Call Bedrock Converse Stream endpoint

For streaming responses, use the `/converse-stream` endpoint:

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/my-claude-model/converse-stream' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Tell me a short story"}]
        }
    ],
    "inferenceConfig": {
        "temperature": 0.7,
        "maxTokens": 200
    }
}'
```

### Supported Bedrock Endpoints with config.yaml

When using models from config.yaml, you can call any Bedrock endpoint:

| Endpoint | Description | Example |
|----------|-------------|---------|
| `/model/{model_name}/converse` | Converse API | `http://0.0.0.0:4000/bedrock/model/my-claude-model/converse` |
| `/model/{model_name}/converse-stream` | Streaming Converse | `http://0.0.0.0:4000/bedrock/model/my-claude-model/converse-stream` |
| `/model/{model_name}/invoke` | Legacy Invoke API | `http://0.0.0.0:4000/bedrock/model/my-claude-model/invoke` |
| `/model/{model_name}/invoke-with-response-stream` | Legacy Streaming | `http://0.0.0.0:4000/bedrock/model/my-claude-model/invoke-with-response-stream` |

The proxy automatically resolves the `model_name` to the actual Bedrock model ID and region configured in your `config.yaml`.

### Load Balancing Across Multiple Deployments

Define multiple Bedrock deployments with the same `model_name` to enable automatic load balancing.

#### 1. Define multiple deployments in config.yaml

```yaml showLineNumbers
model_list:
  # First deployment - us-west-2
  - model_name: my-claude-model
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_region_name: us-west-2
      custom_llm_provider: bedrock
  
  # Second deployment - us-east-1 (load balanced)
  - model_name: my-claude-model
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_region_name: us-east-1
      custom_llm_provider: bedrock
```

#### 2. Start proxy with config

```bash showLineNumbers
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Call the endpoint - requests are automatically load balanced

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/my-claude-model/invoke' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "max_tokens": 100,
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "anthropic_version": "bedrock-2023-05-31"
}'
```

The proxy will automatically distribute requests across both `us-west-2` and `us-east-1` deployments. This works for all Bedrock endpoints: `/invoke`, `/invoke-with-response-stream`, `/converse`, and `/converse-stream`.

#### Using boto3 SDK with load balancing

You can also call the load-balanced endpoint using the boto3 SDK:

```python showLineNumbers
import boto3
import json
import os

# Set dummy AWS credentials (required by boto3, but not used by LiteLLM proxy)
os.environ['AWS_ACCESS_KEY_ID'] = 'dummy'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'dummy'
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = "sk-1234"  # your litellm proxy api key

# Point boto3 to the LiteLLM proxy
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    endpoint_url='http://0.0.0.0:4000/bedrock'
)

# Call the load-balanced model
response = bedrock_runtime.invoke_model(
    modelId='my-claude-model',  # Your model_name from config.yaml
    contentType='application/json',
    accept='application/json',
    body=json.dumps({
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    })
)

# Parse response
response_body = json.loads(response['body'].read())
print(response_body['content'][0]['text'])
```

The proxy will automatically load balance your boto3 requests across all configured deployments.


## Examples

Anything after `http://0.0.0.0:4000/bedrock` is treated as a provider-specific route, and handled accordingly.

Key Changes: 

| **Original Endpoint**                                | **Replace With**                  |
|------------------------------------------------------|-----------------------------------|
| `https://bedrock-runtime.{aws_region_name}.amazonaws.com`          | `http://0.0.0.0:4000/bedrock` (LITELLM_PROXY_BASE_URL="http://0.0.0.0:4000")      |
| `AWS4-HMAC-SHA256..`                                 | `Bearer anything` (use `Bearer LITELLM_VIRTUAL_KEY` if Virtual Keys are setup on proxy)                    |



### **Example 1: Converse API**

#### LiteLLM Proxy Call 

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/cohere.command-r-v1:0/converse' \
-H 'Authorization: Bearer sk-anything' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
         {"role": "user",
        "content": [{"text": "Hello"}]
    }
    ]
}'
```

#### Direct Bedrock API Call 

```bash showLineNumbers
curl -X POST 'https://bedrock-runtime.us-west-2.amazonaws.com/model/cohere.command-r-v1:0/converse' \
-H 'Authorization: AWS4-HMAC-SHA256..' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
         {"role": "user",
        "content": [{"text": "Hello"}]
    }
    ]
}'
```

### **Example 2: Apply Guardrail**

**Setup**: Set AWS credentials for direct passthrough

```bash showLineNumbers
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION_NAME="us-west-2"
```

Start proxy:

```bash showLineNumbers
litellm

# RUNNING on http://0.0.0.0:4000
```

#### LiteLLM Proxy Call 

```bash showLineNumbers
curl "http://0.0.0.0:4000/bedrock/guardrail/guardrailIdentifier/version/guardrailVersion/apply" \
    -H 'Authorization: Bearer sk-anything' \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{"text": {"text": "Hello world"}}],
      "source": "INPUT"
       }'
```

#### Direct Bedrock API Call

```bash showLineNumbers
curl "https://bedrock-runtime.us-west-2.amazonaws.com/guardrail/guardrailIdentifier/version/guardrailVersion/apply" \
    -H 'Authorization: AWS4-HMAC-SHA256..' \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{"text": {"text": "Hello world"}}],
      "source": "INPUT"
       }'
```

### **Example 3: Query Knowledge Base**

**Setup**: Set AWS credentials for direct passthrough

```bash showLineNumbers
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION_NAME="us-west-2"
```

Start proxy:

```bash showLineNumbers
litellm

# RUNNING on http://0.0.0.0:4000
```

#### LiteLLM Proxy Call

```bash showLineNumbers
curl -X POST "http://0.0.0.0:4000/bedrock/knowledgebases/{knowledgeBaseId}/retrieve" \
-H 'Authorization: Bearer sk-anything' \
-H 'Content-Type: application/json' \
-d '{
    "nextToken": "string",
    "retrievalConfiguration": { 
        "vectorSearchConfiguration": { 
          "filter": { ... },
          "numberOfResults": number,
          "overrideSearchType": "string"
        }
    },
    "retrievalQuery": { 
        "text": "string"
    }
}'
```

#### Direct Bedrock API Call 

```bash showLineNumbers
curl -X POST "https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases/{knowledgeBaseId}/retrieve" \
-H 'Authorization: AWS4-HMAC-SHA256..' \
-H 'Content-Type: application/json' \
-d '{
    "nextToken": "string",
    "retrievalConfiguration": { 
        "vectorSearchConfiguration": { 
          "filter": { ... },
          "numberOfResults": number,
          "overrideSearchType": "string"
        }
    },
    "retrievalQuery": { 
        "text": "string"
    }
}'
```


## Advanced - Use with Virtual Keys 

Pre-requisites
- [Setup proxy with DB](../proxy/virtual_keys.md#setup)

Use this, to avoid giving developers the raw AWS Keys, but still letting them use AWS Bedrock endpoints.

### Usage

1. Setup environment

```bash showLineNumbers
export DATABASE_URL=""
export LITELLM_MASTER_KEY=""
export AWS_ACCESS_KEY_ID=""  # Access key
export AWS_SECRET_ACCESS_KEY="" # Secret access key
export AWS_REGION_NAME="" # us-east-1, us-east-2, us-west-1, us-west-2
```

```bash showLineNumbers
litellm

# RUNNING on http://0.0.0.0:4000
```

2. Generate virtual key 

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{}'
```

Expected Response 

```bash showLineNumbers
{
    ...
    "key": "sk-1234ewknldferwedojwojw"
}
```

3. Test it! 


```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/bedrock/model/cohere.command-r-v1:0/converse' \
-H 'Authorization: Bearer sk-1234ewknldferwedojwojw' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
         {"role": "user",
        "content": [{"text": "Hello"}]
    }
    ]
}'
```

## Advanced - Bedrock Agents 

Call Bedrock Agents via LiteLLM proxy

**Setup**: Set AWS credentials on your LiteLLM proxy server

```bash showLineNumbers
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION_NAME="us-west-2"
```

Start proxy:

```bash showLineNumbers
litellm

# RUNNING on http://0.0.0.0:4000
```

**Usage from Python**:

```python showLineNumbers
import os 
import boto3

# Set dummy AWS credentials (required by boto3, but not used by LiteLLM proxy)
os.environ["AWS_ACCESS_KEY_ID"] = "dummy"
os.environ["AWS_SECRET_ACCESS_KEY"] = "dummy"
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "sk-1234"  # your litellm proxy api key

# Create the client
runtime_client = boto3.client(
    service_name="bedrock-agent-runtime", 
    region_name="us-west-2", 
    endpoint_url="http://0.0.0.0:4000/bedrock"
)

response = runtime_client.invoke_agent(
    agentId="L1RT58GYRW",
    agentAliasId="MFPSBCXYTW",
    sessionId="12345",
    inputText="Who do you know?"
)

completion = ""

for event in response.get("completion"):
    chunk = event["chunk"]
    completion += chunk["bytes"].decode()

print(completion)
```

## Using LangChain AWS SDK with LiteLLM

You can use the [LangChain AWS SDK](https://python.langchain.com/docs/integrations/chat/bedrock/) with LiteLLM Proxy to get cost tracking, load balancing, and other LiteLLM features.

### Quick Start

**1. Install LangChain AWS**:

```bash showLineNumbers
pip install langchain-aws
```

**2. Setup LiteLLM Proxy**:

Create a `config.yaml`:

```yaml showLineNumbers
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0
      aws_region_name: us-east-1
      custom_llm_provider: bedrock
```

Start the proxy:

```bash showLineNumbers
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

**3. Use LangChain with LiteLLM**:

```python showLineNumbers
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

# Your LiteLLM API key
API_KEY = "Bearer sk-1234"

# Initialize ChatBedrockConverse pointing to LiteLLM proxy
llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    endpoint_url="http://localhost:4000/bedrock",
    region_name="us-east-1",
    aws_access_key_id=API_KEY,
    aws_secret_access_key="bedrock"  # Any non-empty value works
)

# Invoke the model
messages = [HumanMessage(content="Hello, how are you?")]
response = llm.invoke(messages)

print(response.content)
```

### Advanced Example: PDF Document Processing with Citations

LangChain AWS SDK supports Bedrock's document processing features. Here's how to use it with LiteLLM:

```python showLineNumbers
import os
import json
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

# Your LiteLLM API key
API_KEY = "Bearer sk-1234"

def get_llm() -> ChatBedrockConverse:
    """Initialize LLM pointing to LiteLLM proxy"""
    llm = ChatBedrockConverse(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        base_model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        endpoint_url="http://localhost:4000/bedrock",
        region_name="us-east-1",
        aws_access_key_id=API_KEY,
        aws_secret_access_key="bedrock"
    )
    return llm

if __name__ == "__main__":
    # Initialize the LLM
    llm = get_llm()
    
    # Read PDF file as bytes (Converse API requires raw bytes)
    with open("your-document.pdf", "rb") as file:
        file_bytes = file.read()
    
    # Prepare messages with document attachment
    messages = [
        HumanMessage(content=[
            {"text": "What is the policy number in this document?"},
            {
                "document": {
                    "format": "pdf",
                    "name": "PolicyDocument",
                    "source": {"bytes": file_bytes},
                    "citations": {"enabled": True}
                }
            }
        ])
    ]
    
    # Invoke the LLM
    response = llm.invoke(messages)
    
    # Print response with citations
    print(json.dumps(response.content, indent=4))
```

### Supported LangChain Features

All LangChain AWS features work with LiteLLM:

| Feature | Supported | Notes |
|---------|-----------|-------|
| Text Generation | ✅ | Full support |
| Streaming | ✅ | Use `stream()` method |
| Document Processing | ✅ | PDF, images, etc. |
| Citations | ✅ | Enable in document config |
| Tool Use | ✅ | Function calling support |
| Multi-modal | ✅ | Text + images + documents |

### Troubleshooting

**Issue**: `UnknownOperationException` error

**Solution**: Make sure you're using the correct endpoint URL format:
- ✅ Correct: `http://localhost:4000/bedrock`
- ❌ Wrong: `http://localhost:4000/bedrock/v2`

**Issue**: Authentication errors

**Solution**: Ensure your API key is in the correct format:
```python
aws_access_key_id="Bearer sk-1234"  # Include "Bearer " prefix
```
