import json
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

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
        == "https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases/..%2F..%2Fknowledgebases%2Fother"
        "%3Fx%3D1%23frag/retrieve"
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
    assert "userContext" not in body


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


def _search_body(extra_body: dict[str, object] | None, litellm_params: dict[str, object]) -> dict[str, object]:
    config: Final = BedrockVectorStoreConfig()
    mock_log: Final = MagicMock()
    mock_log.model_call_details = {}
    _, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={"max_num_results": 3},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params=litellm_params,
        extra_body=extra_body,
    )
    return body


def test_transform_search_request_forwards_user_context_from_extra_body():
    body = _search_body(extra_body={"userContext": {"userId": "alice@example.com"}}, litellm_params={})

    assert body["userContext"] == {"userId": "alice@example.com"}
    assert body["retrievalConfiguration"] == {"vectorSearchConfiguration": {"numberOfResults": 3}}


def test_transform_search_request_forwards_top_level_user_context_from_litellm_params():
    body = _search_body(
        extra_body=None,
        litellm_params={"vector_store_id": "kb123", "user_context": {"userId": "bob@example.com"}},
    )

    assert body["userContext"] == {"userId": "bob@example.com"}


def test_transform_search_request_prefers_extra_body_user_context_over_top_level():
    body = _search_body(
        extra_body={"userContext": {"userId": "alice@example.com"}},
        litellm_params={"userContext": {"userId": "bob@example.com"}},
    )

    assert body["userContext"] == {"userId": "alice@example.com"}


def test_get_file_id_from_metadata():
    """Test that file_id is correctly extracted from metadata."""
    config = BedrockVectorStoreConfig()

    metadata_with_uri = {
        "x-amz-bedrock-kb-source-uri": "https://www.litellm.ai",
        "x-amz-bedrock-kb-chunk-id": "1%3A0%3AjNYPg5YByRuP5PdK96co",
    }
    file_id = config._get_file_id_from_metadata(metadata_with_uri)
    assert file_id == "https://www.litellm.ai"

    metadata_without_uri = {"x-amz-bedrock-kb-chunk-id": "1%3A0%3AjNYPg5YByRuP5PdK96co"}
    file_id = config._get_file_id_from_metadata(metadata_without_uri)
    assert file_id == "bedrock-kb-1%3A0%3AjNYPg5YByRuP5PdK96co"

    file_id = config._get_file_id_from_metadata({})
    assert file_id == "bedrock-kb-unknown"


def test_get_filename_from_metadata():
    """Test that filename is correctly extracted from metadata."""
    config = BedrockVectorStoreConfig()

    metadata_with_path = {"x-amz-bedrock-kb-source-uri": "https://docs.litellm.ai/tutorial/setup.html"}
    filename = config._get_filename_from_metadata(metadata_with_path)
    assert filename == "setup.html"

    metadata_domain_only = {"x-amz-bedrock-kb-source-uri": "https://www.litellm.ai"}
    filename = config._get_filename_from_metadata(metadata_domain_only)
    assert filename == "www.litellm.ai"

    metadata_without_uri = {"x-amz-bedrock-kb-data-source-id": "CCEJIRXXFI"}
    filename = config._get_filename_from_metadata(metadata_without_uri)
    assert filename == "bedrock-kb-document-CCEJIRXXFI"

    filename = config._get_filename_from_metadata({})
    assert filename == "bedrock-kb-document-unknown"


def test_get_attributes_from_metadata():
    """Test that attributes are correctly extracted from metadata."""
    config = BedrockVectorStoreConfig()

    metadata = {
        "x-amz-bedrock-kb-source-uri": "https://www.litellm.ai",
        "x-amz-bedrock-kb-chunk-id": "1%3A0%3AjNYPg5YByRuP5PdK96co",
        "x-amz-bedrock-kb-data-source-id": "CCEJIRXXFI",
    }
    attributes = config._get_attributes_from_metadata(metadata)
    assert attributes == metadata
    assert attributes is not metadata

    attributes = config._get_attributes_from_metadata({})
    assert attributes == {}

    attributes = config._get_attributes_from_metadata(None)
    assert attributes == {}


BEDROCK_RETRIEVE_URL = "https://bedrock-agent-runtime.us-east-1.amazonaws.com/knowledgebases/T37J8R4WTM/retrieve"
BEDROCK_RETRIEVE_RESPONSE = {
    "retrievalResults": [
        {
            "content": {"text": "Mock search result"},
            "location": {"type": "S3", "s3Location": {"uri": "s3://docs/setup.html"}},
            "metadata": {
                "x-amz-bedrock-kb-source-uri": "https://docs.litellm.ai/setup.html",
                "x-amz-bedrock-kb-chunk-id": "chunk-1",
            },
            "score": 0.9,
        }
    ]
}


@pytest.mark.asyncio
async def test_bedrock_search_with_router(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    import litellm
    from litellm.router import Router

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    retrieve_route = respx_mock.post(BEDROCK_RETRIEVE_URL).mock(
        return_value=httpx.Response(200, json=BEDROCK_RETRIEVE_RESPONSE)
    )

    _router = Router(model_list=[])
    search_response = await _router.avector_store_search(
        query="what happens after we add a model",
        vector_store_id="T37J8R4WTM",
        custom_llm_provider="bedrock",
        aws_access_key_id="test_access_key",
        aws_secret_access_key="test_secret_key",
        aws_region_name="us-east-1",
    )

    assert retrieve_route.call_count == 1
    request = retrieve_route.calls.last.request
    assert "AWS4-HMAC-SHA256 Credential=test_access_key" in request.headers["authorization"]
    body = json.loads(request.read())
    assert body["retrievalQuery"]["text"] == "what happens after we add a model"

    assert search_response is not None
    assert search_response["object"] == "vector_store.search_results.page"
    assert len(search_response["data"]) == 1
    assert search_response["data"][0]["score"] == 0.9
    assert search_response["data"][0]["content"][0]["text"] == "Mock search result"


@pytest.mark.asyncio
async def test_bedrock_search_with_credentials_managed_registry(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    """
    Test that the vector store search uses the credential accessor from the registry
    when AWS environment variables are not set, ensuring credentials are managed properly.
    """
    from datetime import datetime, timezone

    import litellm
    from litellm.router import Router
    from litellm.types.utils import CredentialItem
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

    test_credentials = CredentialItem(
        credential_name="bedrock-litellm-website-knowledgebase",
        credential_info={
            "provider": "aws",
            "description": "Test AWS credentials for bedrock",
        },
        credential_values={
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_region_name": "us-east-1",
        },
    )

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "credential_list", [test_credentials])
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)

    vector_store = LiteLLM_ManagedVectorStore(
        vector_store_id="T37J8R4WTM",
        custom_llm_provider="bedrock",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        litellm_credential_name="bedrock-litellm-website-knowledgebase",
    )

    registry = VectorStoreRegistry([vector_store])
    monkeypatch.setattr(litellm, "vector_store_registry", registry)

    retrieve_route = respx_mock.post(BEDROCK_RETRIEVE_URL).mock(
        return_value=httpx.Response(200, json=BEDROCK_RETRIEVE_RESPONSE)
    )

    _router = Router(model_list=[])
    search_response = await _router.avector_store_search(
        query="what happens after we add a model",
        vector_store_id="T37J8R4WTM",
        custom_llm_provider="bedrock",
    )

    assert retrieve_route.call_count == 1
    request = retrieve_route.calls.last.request
    assert "AWS4-HMAC-SHA256 Credential=test_access_key" in request.headers["authorization"]

    assert search_response["data"][0]["score"] == 0.9
    assert search_response["data"][0]["content"][0]["text"] == "Mock search result"
