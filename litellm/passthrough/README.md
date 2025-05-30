This makes it easier to pass through requests to the LLM APIs.

E.g. Route to VLLM's `/classify` endpoint:


## SDK (Basic)

```python
import litellm


response = litellm.llm_passthrough_route(
    model="papluca/xlm-roberta-base-language-detection",
    input="Hello, world!",
    api_base="http://localhost:8090",
    endpoint="classify",
)

print(response)
```

## SDK (Router)

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "papluca/xlm-roberta-base-language-detection",
            "litellm_params": {
                "model": "hosted_vllm/papluca/xlm-roberta-base-language-detection",
                "api_base": "http://localhost:8090", 
            }
        }
    ]
)

request_data = {
    "model": "papluca/xlm-roberta-base-language-detection",
    "input": "Hello, world!",
    "endpoint": "classify",
}

response = router.llm_passthrough_route(**request_data)

print(response)
```

## PROXY 

1. Setup config.yaml 

```yaml
model_list:
  - model_name: papluca/xlm-roberta-base-language-detection
    litellm_params:
      model: hosted_vllm/papluca/xlm-roberta-base-language-detection
      api_base: http://localhost:8090
```

2. Run the proxy

```bash
litellm proxy --config config.yaml
```

3. Use the proxy

```bash
curl -X POST http://localhost:8000/vllm/classify \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <your-api-key>" \
-d '{"model": "papluca/xlm-roberta-base-language-detection", "input": "Hello, world!"}' \
```