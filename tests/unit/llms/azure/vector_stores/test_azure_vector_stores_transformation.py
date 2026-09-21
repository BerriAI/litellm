from litellm.llms.azure.vector_stores.transformation import AzureOpenAIVectorStoreConfig


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
