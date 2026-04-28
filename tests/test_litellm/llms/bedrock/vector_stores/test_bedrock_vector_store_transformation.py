from unittest.mock import MagicMock

from litellm.llms.bedrock.vector_stores.transformation import BedrockVectorStoreConfig


def test_transform_search_request():
    """
    Test that BedrockVectorStoreConfig correctly transforms search vector store requests.

    Verifies that the transformation creates the proper URL endpoint and request body
    with the expected retrievalQuery structure.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}

    url, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={},
        extra_body=None,
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
    )

    assert url.endswith("/kb123/retrieve")
    assert body["retrievalQuery"].get("text") == "hello"


def test_transform_search_request_uses_only_retrieval_config_from_extra_body():
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}

    url, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={},
        extra_body={
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "overrideSearchType": "HYBRID",
                    "numberOfResults": 8,
                }
            },
            "unrelatedField": {"should_not": "be_forwarded"},
        },
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
    )

    assert url.endswith("/kb123/retrieve")
    assert body["retrievalQuery"].get("text") == "hello"
    assert (
        body["retrievalConfiguration"]["vectorSearchConfiguration"][
            "overrideSearchType"
        ]
        == "HYBRID"
    )
    assert "unrelatedField" not in body
