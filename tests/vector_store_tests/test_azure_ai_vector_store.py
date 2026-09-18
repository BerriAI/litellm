import json
import os
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.llms.azure_ai.vector_stores.transformation import AzureAIVectorStoreConfig
from litellm.types.utils import EmbeddingResponse
from litellm.vector_stores import (
    asearch as vector_store_asearch,
)
from litellm.vector_stores import (
    search as vector_store_search,
)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_basic_search_vector_store(sync_mode):
    litellm._turn_on_debug()
    litellm.set_verbose = True
    base_request_args = {
        "vector_store_id": "my-vector-index",
        "custom_llm_provider": "azure_ai",
        "azure_search_service_name": "azure-kb-search",
        "litellm_embedding_model": "azure_ai/text-embedding-3-large",
        "litellm_embedding_config": {
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_key": os.getenv("AZURE_AI_API_KEY"),
        },
        "api_key": os.getenv("AZURE_SEARCH_API_KEY"),
    }
    default_query = base_request_args.pop("query", "Basic ping")
    print(f"base_request_args: {base_request_args}")
    try:
        if sync_mode:
            response = vector_store_search(query=default_query, **base_request_args)
        else:
            response = await vector_store_asearch(query=default_query, **base_request_args)
    except litellm.InternalServerError:
        pytest.skip("Skipping test due to litellm.InternalServerError")

    print("litellm response=", json.dumps(response, indent=4, default=str))


class RecordingEmbeddingExecutor:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def embed(self, model, query, configuration):
        self.calls.append((model, query, dict(configuration)))
        return self.response

    async def aembed(self, model, query, configuration):
        self.calls.append((model, query, dict(configuration)))
        return self.response


ALIAS_QUERY_VECTOR = [0.5, -0.25, 0.125]
ALIAS_EMBEDDING_RESPONSE = EmbeddingResponse(
    data=[{"embedding": ALIAS_QUERY_VECTOR, "index": 0, "object": "embedding"}]
)
STORE_EMBEDDINGS_URL = "https://embedding.example/v1/embeddings"


def _transform_kwargs(executor):
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return {
        "vector_store_id": "my-vector-index",
        "query": "what is azure search?",
        "vector_store_search_optional_params": {"top_k": 2},
        "api_base": "https://azure-kb-search.search.windows.net",
        "litellm_logging_obj": logging_obj,
        "litellm_params": {
            "litellm_embedding_model": "multilingual-e5-large",
            "azure_search_vector_field": "embedding",
        },
        "embedding_executor": executor,
    }


@pytest.mark.asyncio
async def test_transform_uses_injected_executor_without_embedding_config(respx_mock: respx.MockRouter):
    executor = RecordingEmbeddingExecutor(ALIAS_EMBEDDING_RESPONSE)
    config = AzureAIVectorStoreConfig()
    transform_kwargs = _transform_kwargs(executor)

    url, sync_body = config.transform_search_vector_store_request(**transform_kwargs)
    _, async_body = await config.atransform_search_vector_store_request(**transform_kwargs)

    assert respx_mock.calls.call_count == 0
    assert executor.calls == [("multilingual-e5-large", "what is azure search?", {})] * 2
    assert (
        url == "https://azure-kb-search.search.windows.net/indexes/my-vector-index/docs/search?api-version=2024-07-01"
    )
    assert sync_body == async_body
    assert sync_body["vectorQueries"] == [
        {"vector": ALIAS_QUERY_VECTOR, "fields": "embedding", "kind": "vector", "k": 2}
    ]
    assert sync_body["top"] == 2
    logging_details = transform_kwargs["litellm_logging_obj"].model_call_details
    assert logging_details["embedding_model"] == "multilingual-e5-large"
    assert logging_details["top_k"] == 2


def test_transform_falls_back_to_sdk_embedding_without_executor(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding_route = respx_mock.post(STORE_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": ALIAS_QUERY_VECTOR}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )
    transform_kwargs = _transform_kwargs(None)
    transform_kwargs["litellm_params"] = {
        "litellm_embedding_model": "openai/text-embedding-3-small",
        "litellm_embedding_config": {"api_base": "https://embedding.example/v1", "api_key": "store-key"},
    }

    _, body = AzureAIVectorStoreConfig().transform_search_vector_store_request(**transform_kwargs)

    embedding_request = embedding_route.calls.last.request
    assert embedding_request.headers["authorization"] == "Bearer store-key"
    assert json.loads(embedding_request.read())["input"] == ["what is azure search?"]
    assert body["vectorQueries"][0]["vector"] == ALIAS_QUERY_VECTOR
    assert body["vectorQueries"][0]["fields"] == "contentVector"


def test_transform_requires_embedding_model():
    transform_kwargs = _transform_kwargs(RecordingEmbeddingExecutor(ALIAS_EMBEDDING_RESPONSE))
    transform_kwargs["litellm_params"] = {"litellm_embedding_config": {"api_key": "store-key"}}

    with pytest.raises(ValueError, match="litellm_embedding_model is required"):
        AzureAIVectorStoreConfig().transform_search_vector_store_request(**transform_kwargs)
