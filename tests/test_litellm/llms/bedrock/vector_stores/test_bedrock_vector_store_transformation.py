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
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
        extra_body=None,
    )

    assert url.endswith("/kb123/retrieve")
    assert body["retrievalQuery"].get("text") == "hello"


def test_transform_search_request_encodes_vector_store_id():
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}

    url, body = config.transform_search_vector_store_request(
        vector_store_id="../../knowledgebases/other?x=1#frag",
        query="hello",
        vector_store_search_optional_params={},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
        extra_body=None,
    )

    assert (
        url
        == "https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases/..%2F..%2Fknowledgebases%2Fother%3Fx%3D1%23frag/retrieve"
    )
    assert body["retrievalQuery"].get("text") == "hello"


def test_transform_search_request_uses_only_retrieval_config_from_extra_body():
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}

    url, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
        extra_body={
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "overrideSearchType": "HYBRID",
                    "numberOfResults": 8,
                }
            },
            "unrelatedField": {"should_not": "be_forwarded"},
        },
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


def test_transform_search_request_does_not_mutate_extra_body_and_overrides_number_of_results():
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}
    extra_body = {
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "overrideSearchType": "HYBRID",
                "numberOfResults": 8,
            }
        }
    }

    _, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={"max_num_results": 10},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
        extra_body=extra_body,
    )

    assert (
        body["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"]
        == 10
    )
    assert (
        extra_body["retrievalConfiguration"]["vectorSearchConfiguration"][
            "numberOfResults"
        ]
        == 8
    )


def test_transform_search_request_overrides_filter_without_mutating_extra_body():
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}
    extra_body = {
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "filter": {"equals": {"key": "tenant", "value": "a"}}
            }
        }
    }
    new_filter = {"equals": {"key": "tenant", "value": "b"}}

    _, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={"filters": new_filter},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
        extra_body=extra_body,
    )

    assert (
        body["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"]
        == new_filter
    )
    assert (
        extra_body["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"][
            "equals"
        ]["value"]
        == "a"
    )
