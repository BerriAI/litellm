# Google AI Studio

Pass-through endpoints for Google AI Studio - call provider-specific endpoint, in native format (no translation).

Just replace `https://generativelanguage.googleapis.com` with `LITELLM_PROXY_BASE_URL/gemini` 🚀

#### **Example Usage**
```bash
http://0.0.0.0:4000/gemini/v1beta/models/gemini-1.5-flash:countTokens?key=sk-anything' \
-H 'Content-Type: application/json' \
-d '{
    "contents": [{
        "parts":[{
          "text": "The quick brown fox jumps over the lazy dog."
          }]
        }]
}'
```

Supports **ALL** Google AI Studio Endpoints (including streaming).

[**See All Google AI Studio Endpoints**](https://ai.google.dev/api)

## Quick Start

Let's call the Gemini [`/countTokens` endpoint](https://ai.google.dev/api/tokens#method:-models.counttokens)

1. Add Gemini API Key to your environment 

```bash
export GEMINI_API_KEY=""
```

2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Google AI Studio token counting endpoint

```bash
http://0.0.0.0:4000/gemini/v1beta/models/gemini-1.5-flash:countTokens?key=anything' \
-H 'Content-Type: application/json' \
-d '{
    "contents": [{
        "parts":[{
          "text": "The quick brown fox jumps over the lazy dog."
          }]
        }]
}'
```


## Examples

Anything after `http://0.0.0.0:4000/gemini` is treated as a provider-specific route, and handled accordingly.

Key Changes: 

| **Original Endpoint**                                | **Replace With**                  |
|------------------------------------------------------|-----------------------------------|
| `https://generativelanguage.googleapis.com`          | `http://0.0.0.0:4000/gemini` (LITELLM_PROXY_BASE_URL="http://0.0.0.0:4000")      |
| `key=$GOOGLE_API_KEY`                                 | `key=anything` (use `key=LITELLM_VIRTUAL_KEY` if Virtual Keys are setup on proxy)                    |


### **Example 1: Counting tokens**

#### LiteLLM Proxy Call 

```bash
curl http://0.0.0.0:4000/gemini/v1beta/models/gemini-1.5-flash:countTokens?key=anything \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{
          "text": "The quick brown fox jumps over the lazy dog."
          }],
        }],
      }'
```

#### Direct Google AI Studio Call 

```bash
curl https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:countTokens?key=$GOOGLE_API_KEY \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{
          "text": "The quick brown fox jumps over the lazy dog."
          }],
        }],
      }'
```

### **Example 2: Generate content**

#### LiteLLM Proxy Call 

```bash
curl "http://0.0.0.0:4000/gemini/v1beta/models/gemini-1.5-flash:generateContent?key=anything" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "Write a story about a magic backpack."}]
        }]
       }' 2> /dev/null
```

#### Direct Google AI Studio Call 

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "Write a story about a magic backpack."}]
        }]
       }' 2> /dev/null
```

### **Example 3: Caching**


```bash
curl -X POST "http://0.0.0.0:4000/gemini/v1beta/models/gemini-1.5-flash-001:generateContent?key=anything" \
-H 'Content-Type: application/json' \
-d '{
      "contents": [
        {
          "parts":[{
            "text": "Please summarize this transcript"
          }],
          "role": "user"
        },
      ],
      "cachedContent": "'$CACHE_NAME'"
    }'
```

#### Direct Google AI Studio Call 

```bash
curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
      "contents": [
        {
          "parts":[{
            "text": "Please summarize this transcript"
          }],
          "role": "user"
        },
      ],
      "cachedContent": "'$CACHE_NAME'"
    }'
```


## Advanced - Use with Virtual Keys 

Pre-requisites
- [Setup proxy with DB](../proxy/virtual_keys.md#setup)

Use this, to avoid giving developers the raw Google AI Studio key, but still letting them use Google AI Studio endpoints.

### Usage

1. Setup environment

```bash
export DATABASE_URL=""
export LITELLM_MASTER_KEY=""
export GEMINI_API_KEY=""
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
http://0.0.0.0:4000/gemini/v1beta/models/gemini-1.5-flash:countTokens?key=sk-1234ewknldferwedojwojw' \
-H 'Content-Type: application/json' \
-d '{
    "contents": [{
        "parts":[{
          "text": "The quick brown fox jumps over the lazy dog."
          }]
        }]
}'
```