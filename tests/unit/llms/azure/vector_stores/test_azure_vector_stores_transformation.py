from unittest.mock import Mock

from litellm.llms.azure.vector_stores.transformation import AzureOpenAIVectorStoreConfig
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig


def test_transform_search_vector_store_request_preserves_azure_query_string():
    config = AzureOpenAIVectorStoreConfig()
    api_base = config.get_complete_url(
        api_base="https://x.openai.azure.com",
        litellm_params={"api_version": "2024-10-21"},
    )

    url, _ = config.transform_search_vector_store_request(
        vector_store_id="vs_1",
        query="hello",
        vector_store_search_optional_params={},
        api_base=api_base,
        litellm_logging_obj=None,
        litellm_params={"api_version": "2024-10-21"},
    )

    assert url == "https://x.openai.azure.com/openai/vector_stores/vs_1/search?api-version=2024-10-21"


def test_get_vector_store_url_puts_id_before_api_version_query():
    config = AzureOpenAIVectorStoreConfig()
    api_base = config.get_complete_url(
        api_base="https://x.openai.azure.com",
        litellm_params={"api_version": "2025-04-01-preview"},
    )

    url = config.get_vector_store_url(api_base, "vs_abc")

    assert url == "https://x.openai.azure.com/openai/vector_stores/vs_abc?api-version=2025-04-01-preview"


def test_get_vector_store_url_without_query_is_plain_path():
    url = AzureOpenAIVectorStoreConfig().get_vector_store_url(
        "https://x.openai.azure.com/openai/vector_stores", "vs_abc"
    )

    assert url == "https://x.openai.azure.com/openai/vector_stores/vs_abc"


def test_base_get_vector_store_url_appends_id_for_providers_without_query_string():
    url = BaseVectorStoreConfig.get_vector_store_url(Mock(), "https://api.example.test/v1/vector_stores", "vs_abc")

    assert url == "https://api.example.test/v1/vector_stores/vs_abc"
