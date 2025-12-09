import pytest
import litellm
import json
import os
from litellm.vector_stores import (
    search as vector_store_search,
    asearch as vector_store_asearch,
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
        "litellm_embedding_model": "azure/text-embedding-3-large",
        "litellm_embedding_config": {
            "api_base": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_BASE"),
            "api_key": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_KEY"),
        },
        "api_key": os.getenv("AZURE_SEARCH_API_KEY"),
    }
    default_query = base_request_args.pop("query", "Basic ping")
    print(f"base_request_args: {base_request_args}")
    try:
        if sync_mode:
            response = vector_store_search(query=default_query, **base_request_args)
        else:
            response = await vector_store_asearch(
                query=default_query, **base_request_args
            )
    except litellm.InternalServerError:
        pytest.skip("Skipping test due to litellm.InternalServerError")

    print("litellm response=", json.dumps(response, indent=4, default=str))
