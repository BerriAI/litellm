# VLLM

Pass-through endpoints for VLLM - call provider-specific endpoint, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ❌ | [Tell us if you need this](https://github.com/BerriAI/litellm/issues/new) |
| Streaming | ✅ | |

Just replace `https://my-vllm-server.com` with `LITELLM_PROXY_BASE_URL/vllm` 🚀

#### **Example Usage**

```bash
curl -L -X GET 'http://0.0.0.0:4000/vllm/metrics' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
```

Supports **ALL** VLLM Endpoints (including streaming).

## Quick Start

Let's call the VLLM [`/score` endpoint](https://vllm.readthedocs.io/en/latest/api_reference/api_reference.html)

1. Add a VLLM hosted model to your LiteLLM Proxy 

:::info

Works with LiteLLM v1.72.0+. 

:::

```yaml
model_list:
  - model_name: "my-vllm-model"
    litellm_params:
      model: hosted_vllm/vllm-1.72
      api_base: https://my-vllm-server.com
```

2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the VLLM `/score` endpoint

```bash
curl -X 'POST' \
  'http://0.0.0.0:4000/vllm/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "my-vllm-model",
  "encoding_format": "float",
  "text_1": "What is the capital of France?",
  "text_2": "The capital of France is Paris."
}'
```


## Examples

Anything after `http://0.0.0.0:4000/vllm` is treated as a provider-specific route, and handled accordingly.

Key Changes: 

| **Original Endpoint**                                | **Replace With**                  |
|------------------------------------------------------|-----------------------------------|
| `https://my-vllm-server.com`          | `http://0.0.0.0:4000/vllm` (LITELLM_PROXY_BASE_URL="http://0.0.0.0:4000")      |
| `bearer $VLLM_API_KEY`                                 | `bearer anything` (use `bearer LITELLM_VIRTUAL_KEY` if Virtual Keys are setup on proxy)                    |


### **Example 1: Metrics endpoint**

#### LiteLLM Proxy Call 

```bash
curl -L -X GET 'http://0.0.0.0:4000/vllm/metrics' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_VIRTUAL_KEY' \
```


#### Direct VLLM API Call 

```bash
curl -L -X GET 'https://my-vllm-server.com/metrics' \
-H 'Content-Type: application/json' \
```

### **Example 2: Chat API**

#### LiteLLM Proxy Call 

```bash
curl -L -X POST 'http://0.0.0.0:4000/vllm/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_VIRTUAL_KEY' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.1,
    "model": "qwen2.5-7b-instruct",
}'
```

#### Direct VLLM API Call 

```bash
curl -L -X POST 'https://my-vllm-server.com/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.1,
    "model": "qwen2.5-7b-instruct",
}'
```


## Advanced - Use with Virtual Keys 

Pre-requisites
- [Setup proxy with DB](../proxy/virtual_keys.md#setup)

Use this, to avoid giving developers the raw Cohere API key, but still letting them use Cohere endpoints.

### Usage

1. Setup environment

```bash
export DATABASE_URL=""
export LITELLM_MASTER_KEY=""
export HOSTED_VLLM_API_BASE=""
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
curl -L -X POST 'http://0.0.0.0:4000/vllm/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234ewknldferwedojwojw' \
  --data '{
    "messages": [
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.1,
    "model": "qwen2.5-7b-instruct",
}'
```