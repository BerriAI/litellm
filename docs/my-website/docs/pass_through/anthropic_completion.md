import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Anthropic Passthrough

Pass-through endpoints for Anthropic - call provider-specific endpoint, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ | supports all models on `/messages` endpoint |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ✅ | disable prometheus tracking via `litellm.disable_end_user_cost_tracking_prometheus_only`|
| Streaming | ✅ | |

Just replace `https://api.anthropic.com` with `LITELLM_PROXY_BASE_URL/anthropic`

#### **Example Usage**


<Tabs>
<TabItem value="curl" label="curl">

```bash
curl --request POST \
  --url http://0.0.0.0:4000/anthropic/v1/messages \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header "Authorization: bearer sk-anything" \
  --data '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'
```

</TabItem>
<TabItem value="python" label="Anthropic Python SDK">

```python
from anthropic import Anthropic

# Initialize client with proxy base URL
client = Anthropic(
    base_url="http://0.0.0.0:4000/anthropic", # <proxy-base-url>/anthropic
    api_key="sk-anything" # proxy virtual key
)

# Make a completion request
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)

print(response)
```

</TabItem>
</Tabs>

Supports **ALL** Anthropic Endpoints (including streaming).

[**See All Anthropic Endpoints**](https://docs.anthropic.com/en/api/messages)

## Quick Start

Let's call the Anthropic [`/messages` endpoint](https://docs.anthropic.com/en/api/messages)

1. Add Anthropic API Key to your environment 

```bash
export ANTHROPIC_API_KEY=""
```

2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Anthropic /messages endpoint

```bash
curl http://0.0.0.0:4000/anthropic/v1/messages \
     --header "x-api-key: $LITELLM_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
    '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'
```


## Examples

Anything after `http://0.0.0.0:4000/anthropic` is treated as a provider-specific route, and handled accordingly.

Key Changes: 

| **Original Endpoint**                                | **Replace With**                  |
|------------------------------------------------------|-----------------------------------|
| `https://api.anthropic.com`          | `http://0.0.0.0:4000/anthropic` (LITELLM_PROXY_BASE_URL="http://0.0.0.0:4000")      |
| `bearer $ANTHROPIC_API_KEY`                                 | `bearer anything` (use `bearer LITELLM_VIRTUAL_KEY` if Virtual Keys are setup on proxy)                    |
    

### **Example 1: Messages endpoint**

#### LiteLLM Proxy Call 

```bash
curl --request POST \
  --url http://0.0.0.0:4000/anthropic/v1/messages \
  --header "x-api-key: $LITELLM_API_KEY" \
    --header "anthropic-version: 2023-06-01" \
    --header "content-type: application/json" \
  --data '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
  }'
```

#### Direct Anthropic API Call 

```bash
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
    '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'
```

### **Example 2: Token Counting API**

#### LiteLLM Proxy Call 

```bash
curl --request POST \
    --url http://0.0.0.0:4000/anthropic/v1/messages/count_tokens \
    --header "x-api-key: $LITELLM_API_KEY" \
    --header "anthropic-version: 2023-06-01" \
    --header "anthropic-beta: token-counting-2024-11-01" \
    --header "content-type: application/json" \
    --data \
    '{
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'
```

#### Direct Anthropic API Call 

```bash
curl https://api.anthropic.com/v1/messages/count_tokens \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "anthropic-beta: token-counting-2024-11-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
}'
```

### **Example 3: Batch Messages**


#### LiteLLM Proxy Call 

```bash
curl --request POST \
    --url http://0.0.0.0:4000/anthropic/v1/messages/batches \
    --header "x-api-key: $LITELLM_API_KEY" \
    --header "anthropic-version: 2023-06-01" \
    --header "anthropic-beta: message-batches-2024-09-24" \
    --header "content-type: application/json" \
    --data \
'{
    "requests": [
        {
            "custom_id": "my-first-request",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ]
            }
        },
        {
            "custom_id": "my-second-request",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hi again, friend"}
                ]
            }
        }
    ]
}'
```

#### Direct Anthropic API Call 

```bash
curl https://api.anthropic.com/v1/messages/batches \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "anthropic-beta: message-batches-2024-09-24" \
     --header "content-type: application/json" \
     --data \
'{
    "requests": [
        {
            "custom_id": "my-first-request",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ]
            }
        },
        {
            "custom_id": "my-second-request",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hi again, friend"}
                ]
            }
        }
    ]
}'
```


## Advanced

Pre-requisites
- [Setup proxy with DB](../proxy/virtual_keys.md#setup)

Use this, to avoid giving developers the raw Anthropic API key, but still letting them use Anthropic endpoints.

### Use with Virtual Keys 

1. Setup environment

```bash
export DATABASE_URL=""
export LITELLM_MASTER_KEY=""
export COHERE_API_KEY=""
```

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

2. Generate virtual key 

```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{}'
```

Expected Response 

```bash
{
    ...
    "key": "sk-1234ewknldferwedojwojw"
}
```

3. Test it! 


```bash
curl --request POST \
  --url http://0.0.0.0:4000/anthropic/v1/messages \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header "Authorization: bearer sk-1234ewknldferwedojwojw" \
  --data '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
  }'
```


### Send `litellm_metadata` (tags, end-user cost tracking)

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl --request POST \
  --url http://0.0.0.0:4000/anthropic/v1/messages \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header "Authorization: bearer sk-anything" \
  --data '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ],
    "litellm_metadata": {
        "tags": ["test-tag-1", "test-tag-2"], 
        "user": "test-user" # track end-user/customer cost
    }
  }'
```

</TabItem>
<TabItem value="python" label="Anthropic Python SDK">

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://0.0.0.0:4000/anthropic",
    api_key="sk-anything"
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, world"}
    ],
    extra_body={
        "litellm_metadata": {
            "tags": ["test-tag-1", "test-tag-2"], 
            "user": "test-user" # track end-user/customer cost
        }
    }, 
    ## OR## 
    metadata={ # anthropic native param - https://docs.anthropic.com/en/api/messages
        "user_id": "test-user" # track end-user/customer cost
    }

)

print(response)
```

</TabItem>
</Tabs>