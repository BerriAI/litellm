import Image from '@theme/IdealImage';

# ➡️ Create Pass Through Endpoints 

Add pass through routes to LiteLLM Proxy

**Example:** Add a route `/v1/rerank` that forwards requests to `https://api.cohere.com/v1/rerank` through LiteLLM Proxy


💡 This allows making the following Request to LiteLLM Proxy
```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": ["Carson City is the capital city of the American state of Nevada."]
  }'
```

## Tutorial - Pass through Cohere Re-Rank Endpoint

**Step 1** Define pass through routes on [litellm config.yaml](configs.md)

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"                                  # route you want to add to LiteLLM Proxy Server
      target: "https://api.cohere.com/v1/rerank"          # URL this route should forward requests to
      headers:                                            # headers to forward to this URL
        Authorization: "bearer os.environ/COHERE_API_KEY" # (Optional) Auth Header to forward to your Endpoint
        content-type: application/json                    # (Optional) Extra Headers to pass to this endpoint 
        accept: application/json
```

**Step 2** Start Proxy Server in detailed_debug mode

```shell
litellm --config config.yaml --detailed_debug
```
**Step 3** Make Request to pass through endpoint

Here `http://localhost:4000` is your litellm proxy endpoint

```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": ["Carson City is the capital city of the American state of Nevada.",
                  "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                  "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                  "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
                  "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."]
  }'
```


🎉 **Expected Response**

This request got forwarded from LiteLLM Proxy -> Defined Target URL (with headers)

```shell
{
  "id": "37103a5b-8cfb-48d3-87c7-da288bedd429",
  "results": [
    {
      "index": 2,
      "relevance_score": 0.999071
    },
    {
      "index": 4,
      "relevance_score": 0.7867867
    },
    {
      "index": 0,
      "relevance_score": 0.32713068
    }
  ],
  "meta": {
    "api_version": {
      "version": "1"
    },
    "billed_units": {
      "search_units": 1
    }
  }
}
```

## Tutorial - Pass Through Langfuse Requests


**Step 1** Define pass through routes on [litellm config.yaml](configs.md)

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/api/public/ingestion"                                # route you want to add to LiteLLM Proxy Server
      target: "https://us.cloud.langfuse.com/api/public/ingestion" # URL this route should forward 
      headers:
        LANGFUSE_PUBLIC_KEY: "os.environ/LANGFUSE_DEV_PUBLIC_KEY" # your langfuse account public key
        LANGFUSE_SECRET_KEY: "os.environ/LANGFUSE_DEV_SK_KEY"     # your langfuse account secret key
```

**Step 2** Start Proxy Server in detailed_debug mode

```shell
litellm --config config.yaml --detailed_debug
```
**Step 3** Make Request to pass through endpoint

Run this code to make a sample trace 
```python
from langfuse import Langfuse

langfuse = Langfuse(
    host="http://localhost:4000", # your litellm proxy endpoint
    public_key="anything",        # no key required since this is a pass through
    secret_key="anything",        # no key required since this is a pass through
)

print("sending langfuse trace request")
trace = langfuse.trace(name="test-trace-litellm-proxy-passthrough")
print("flushing langfuse request")
langfuse.flush()

print("flushed langfuse request")
```


🎉 **Expected Response**

On success
Expect to see the following Trace Generated on your Langfuse Dashboard

<Image img={require('../../img/proxy_langfuse.png')} />

You will see the following endpoint called on your litellm proxy server logs

```shell
POST /api/public/ingestion HTTP/1.1" 207 Multi-Status
```


