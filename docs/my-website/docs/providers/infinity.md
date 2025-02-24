# Infinity

| Property | Details |
|-------|-------|
| Description | Infinity is a high-throughput, low-latency REST API for serving text-embeddings, reranking models and clip|
| Provider Route on LiteLLM | `infinity/` |
| Supported Operations | `/rerank` |
| Link to Provider Doc | [Infinity ↗](https://github.com/michaelfeil/infinity) |


## **Usage - LiteLLM Python SDK**

```python
from litellm import rerank

response = rerank(
    model="infinity/rerank",
    query="What is the capital of France?",
    documents=["Paris", "London", "Berlin", "Madrid"],
)
```

## **Usage - LiteLLM Proxy**

LiteLLM provides an cohere api compatible `/rerank` endpoint for Rerank calls.

**Setup**

Add this to your litellm proxy config.yaml

```yaml
model_list:
  - model_name: custom-infinity-rerank
    litellm_params:
      model: infinity/rerank
      api_key: os.environ/INFINITY_API_KEY
      api_base: https://localhost:8080
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Test request

```bash
curl http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-infinity-rerank",
    "query": "What is the capital of the United States?",
    "documents": [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country."
    ],
    "top_n": 3
  }'
```


