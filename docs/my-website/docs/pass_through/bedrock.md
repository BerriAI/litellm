# Bedrock SDK

Pass-through endpoints for Bedrock - call provider-specific endpoint, in native format (no translation).

Just replace `https://bedrock-runtime.{aws_region_name}.amazonaws.com` with `LITELLM_PROXY_BASE_URL/bedrock` 🚀

#### **Example Usage**
```bash
curl -X POST 'http://0.0.0.0:4000/bedrock/model/cohere.command-r-v1:0/converse' \
-H 'Authorization: Bearer anything' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
         {"role": "user",
        "content": [{"text": "Hello"}]
    }
    ]
}'
```

Supports **ALL** Bedrock Endpoints (including streaming).

[**See All Bedrock Endpoints**](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)

## Quick Start

Let's call the Bedrock [`/converse` endpoint](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)

1. Add AWS Keyss to your environment 

```bash
export AWS_ACCESS_KEY_ID=""  # Access key
export AWS_SECRET_ACCESS_KEY="" # Secret access key
export AWS_REGION_NAME="" # us-east-1, us-east-2, us-west-1, us-west-2
```

2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Bedrock converse endpoint

```bash
curl -X POST 'http://0.0.0.0:4000/bedrock/model/cohere.command-r-v1:0/converse' \
-H 'Authorization: Bearer anything' \
-H 'Content-Type: application/json' \
-d '{
    "messages": [
         {"role": "user",
        "content": [{"text": "Hello"}]
    }
    ]
}'
```


## Examples

Anything after `http://0.0.0.0:4000/bedrock` is treated as a provider-specific route, and handled accordingly.

Key Changes: 

| **Original Endpoint**                                | **Replace With**                  |
|------------------------------------------------------|-----------------------------------|
| `https://bedrock-runtime.{aws_region_name}.amazonaws.com`          | `http://0.0.0.0:4000/bedrock` (LITELLM_PROXY_BASE_URL="http://0.0.0.0:4000")      |
| `AWS4-HMAC-SHA256..`                                 | `Bearer anything` (use `Bearer LITELLM_VIRTUAL_KEY` if Virtual Keys are setup on proxy)                    |



### **Example 1: Converse API**

#### LiteLLM Proxy Call 

```bash
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

```bash
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

#### LiteLLM Proxy Call 

```bash
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

```bash
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

```bash
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

```bash
curl -X POST "https://bedrock-runtime.us-west-2.amazonaws.com/knowledgebases/{knowledgeBaseId}/retrieve" \
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

```bash
export DATABASE_URL=""
export LITELLM_MASTER_KEY=""
export AWS_ACCESS_KEY_ID=""  # Access key
export AWS_SECRET_ACCESS_KEY="" # Secret access key
export AWS_REGION_NAME="" # us-east-1, us-east-2, us-west-1, us-west-2
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