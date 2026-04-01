# Add Rerank Provider

LiteLLM **follows the Cohere Rerank API format** for all rerank providers. Here's how to add a new rerank provider:

## 1. Create a transformation.py file

Create a config class named `<Provider><Endpoint>Config` that inherits from [`BaseRerankConfig`](https://github.com/BerriAI/litellm/blob/main/litellm/llms/base_llm/rerank/transformation.py):

```python
from litellm.types.rerank import OptionalRerankParams, RerankRequest, RerankResponse
class YourProviderRerankConfig(BaseRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            # ... other supported params
        ]

    def transform_rerank_request(self, model: str, optional_rerank_params: Dict, headers: dict) -> dict:
        # Transform request to RerankRequest spec
        return rerank_request.model_dump(exclude_none=True)

    def transform_rerank_response(self, model: str, raw_response: httpx.Response, ...) -> RerankResponse:
        # Transform provider response to RerankResponse
        return RerankResponse(**raw_response_json)
```


## 2. Register Your Provider
Add your provider to `litellm.utils.get_provider_rerank_config()`:

```python
elif litellm.LlmProviders.YOUR_PROVIDER == provider:
    return litellm.YourProviderRerankConfig()
```


## 3. Add Provider to `rerank_api/main.py`

Add a code block to handle when your provider is called. Your provider should use the `base_llm_http_handler.rerank` method


```python
elif _custom_llm_provider == "your_provider":
    ...
    response = base_llm_http_handler.rerank(
        model=model,
        custom_llm_provider=_custom_llm_provider,
        optional_rerank_params=optional_rerank_params,
        logging_obj=litellm_logging_obj,
        timeout=optional_params.timeout,
        api_key=dynamic_api_key or optional_params.api_key,
        api_base=api_base,
        _is_async=_is_async,
        headers=headers or litellm.headers or {},
        client=client,
        mod el_response=model_response,
    )
    ...
```

## 4. Add Tests

Add a test file to [`tests/llm_translation`](https://github.com/BerriAI/litellm/tree/main/tests/llm_translation)

```python
def test_basic_rerank_cohere():
    response = litellm.rerank(
        model="cohere/rerank-english-v3.0",
        query="hello",
        documents=["hello", "world"],
        top_n=3,
    )

    print("re rank response: ", response)

    assert response.id is not None
    assert response.results is not None
```


## Reference PRs
- [Add Infinity Rerank](https://github.com/BerriAI/litellm/pull/7321)