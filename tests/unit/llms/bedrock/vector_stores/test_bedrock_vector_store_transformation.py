import json
import traceback
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock

import botocore.session
import httpx
import pytest
import respx

import litellm
from litellm.llms.bedrock.common_utils import BedrockError
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


async def _search_with_filters(respx_mock: respx.MockRouter, filters: object, **kwargs: object) -> respx.Route:
    import litellm

    retrieve_route: Final = respx_mock.post(BEDROCK_RETRIEVE_URL).mock(
        return_value=httpx.Response(200, json=BEDROCK_RETRIEVE_RESPONSE)
    )
    await litellm.vector_stores.asearch(
        vector_store_id="T37J8R4WTM",
        query="vacation policy",
        filters=filters,
        custom_llm_provider="bedrock",
        aws_access_key_id="test_access_key",
        aws_secret_access_key="test_secret_key",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return retrieve_route


def _sent_filter(retrieve_route: respx.Route) -> object:
    body: Final = json.loads(retrieve_route.calls.last.request.read())
    return body.get("retrievalConfiguration", {}).get("vectorSearchConfiguration", {}).get("filter")


@pytest.mark.parametrize(
    ("openai_filter", "aws_filter"),
    [
        pytest.param(
            {"type": "eq", "key": "department", "value": "hr"},
            {"equals": {"key": "department", "value": "hr"}},
            id="eq",
        ),
        pytest.param(
            {"type": "ne", "key": "archived", "value": True},
            {"notEquals": {"key": "archived", "value": True}},
            id="ne-bool",
        ),
        pytest.param(
            {"type": "gt", "key": "year", "value": 2022},
            {"greaterThan": {"key": "year", "value": 2022}},
            id="gt-int",
        ),
        pytest.param(
            {"type": "gte", "key": "score", "value": 0.5},
            {"greaterThanOrEquals": {"key": "score", "value": 0.5}},
            id="gte-float",
        ),
        pytest.param(
            {"type": "lt", "key": "year", "value": 2024},
            {"lessThan": {"key": "year", "value": 2024}},
            id="lt",
        ),
        pytest.param(
            {"type": "lte", "key": "year", "value": 2024},
            {"lessThanOrEquals": {"key": "year", "value": 2024}},
            id="lte",
        ),
        pytest.param(
            {"type": "in", "key": "department", "value": ["hr", "sales"]},
            {"in": {"key": "department", "value": ["hr", "sales"]}},
            id="in",
        ),
        pytest.param(
            {"type": "nin", "key": "department", "value": ["hr", "sales"]},
            {"notIn": {"key": "department", "value": ["hr", "sales"]}},
            id="nin",
        ),
        pytest.param(
            {"type": "in", "key": "year", "value": [2023, 2024]},
            {"orAll": [{"equals": {"key": "year", "value": 2023}}, {"equals": {"key": "year", "value": 2024}}]},
            id="in-numbers-as-or-of-equals",
        ),
        pytest.param(
            {"type": "nin", "key": "year", "value": [2022, 2023]},
            {"andAll": [{"notEquals": {"key": "year", "value": 2022}}, {"notEquals": {"key": "year", "value": 2023}}]},
            id="nin-numbers-as-and-of-not-equals",
        ),
        pytest.param(
            {"type": "in", "key": "code", "value": ["hr", 7]},
            {"orAll": [{"equals": {"key": "code", "value": "hr"}}, {"equals": {"key": "code", "value": 7}}]},
            id="in-strings-and-numbers-as-or-of-equals",
        ),
        pytest.param(
            {"type": "in", "key": "year", "value": [2023]},
            {"equals": {"key": "year", "value": 2023}},
            id="in-one-number-unwrapped",
        ),
        pytest.param(
            {
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "department", "value": "hr"},
                    {"type": "gte", "key": "year", "value": 2024},
                ],
            },
            {
                "andAll": [
                    {"equals": {"key": "department", "value": "hr"}},
                    {"greaterThanOrEquals": {"key": "year", "value": 2024}},
                ]
            },
            id="and",
        ),
        pytest.param(
            {
                "type": "or",
                "filters": [
                    {"type": "eq", "key": "department", "value": "sales"},
                    {
                        "type": "and",
                        "filters": [
                            {"type": "eq", "key": "department", "value": "hr"},
                            {"type": "lt", "key": "year", "value": 2024},
                        ],
                    },
                ],
            },
            {
                "orAll": [
                    {"equals": {"key": "department", "value": "sales"}},
                    {
                        "andAll": [
                            {"equals": {"key": "department", "value": "hr"}},
                            {"lessThan": {"key": "year", "value": 2024}},
                        ]
                    },
                ]
            },
            id="or-with-nested-and",
        ),
        pytest.param(
            {"type": "or", "filters": [{"type": "eq", "key": "department", "value": "hr"}]},
            {"equals": {"key": "department", "value": "hr"}},
            id="one-member-compound-unwrapped",
        ),
        pytest.param(
            {"andAll": [{"equals": {"key": "a", "value": 1}}, {"startsWith": {"key": "b", "value": "x"}}]},
            {"andAll": [{"equals": {"key": "a", "value": 1}}, {"startsWith": {"key": "b", "value": "x"}}]},
            id="aws-native-unchanged",
        ),
        pytest.param(
            {"key": "department", "value": "hr", "operator": "ne"},
            {"notEquals": {"key": "department", "value": "hr"}},
            id="operator-shape-still-mapped",
        ),
        pytest.param(
            {"key": "department", "value": "hr", "operator": "eq", "type": "string"},
            {"equals": {"key": "department", "value": "hr"}},
            id="operator-shape-with-type-key-still-mapped",
        ),
        pytest.param(
            {
                "and": [
                    {"key": "department", "value": "hr", "operator": "eq"},
                    {"key": "year", "value": 2024, "operator": "gte"},
                ]
            },
            {
                "andAll": [
                    {"equals": {"key": "department", "value": "hr"}},
                    {"greaterThanOrEquals": {"key": "year", "value": 2024}},
                ]
            },
            id="and-list-shape-still-mapped",
        ),
        pytest.param(
            {
                "type": "and",
                "and": [
                    {"key": "department", "value": "hr", "operator": "eq"},
                    {"key": "year", "value": 2024, "operator": "gte"},
                ],
            },
            {
                "andAll": [
                    {"equals": {"key": "department", "value": "hr"}},
                    {"greaterThanOrEquals": {"key": "year", "value": 2024}},
                ]
            },
            id="and-list-shape-with-type-key-still-mapped",
        ),
    ],
)
@pytest.mark.asyncio
async def test_search_sends_openai_filters_as_aws_retrieval_filters(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    openai_filter: dict[str, object],
    aws_filter: dict[str, object],
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    retrieve_route: Final = await _search_with_filters(respx_mock, openai_filter)

    assert _sent_filter(retrieve_route) == aws_filter


AWS_RETRIEVAL_FILTER_KEYS: Final = tuple(
    botocore.session.get_session().get_service_model("bedrock-agent-runtime").shape_for("RetrievalFilter").members
)


@pytest.mark.parametrize("aws_key", AWS_RETRIEVAL_FILTER_KEYS)
@pytest.mark.asyncio
async def test_search_sends_aws_native_filters_with_a_type_key_unchanged(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, aws_key: str
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    aws_filter: Final = {aws_key: {"key": "department", "value": "hr"}, "type": "eq"}

    retrieve_route: Final = await _search_with_filters(respx_mock, aws_filter)

    assert _sent_filter(retrieve_route) == aws_filter


INVALID_FILTERS: Final = (
    ("unknown-type", {"type": "regex", "key": "department", "value": "h.*"}, "'regex'"),
    ("missing-value", {"type": "eq", "key": "department"}, "filters.eq.value"),
    ("empty-compound", {"type": "and", "filters": []}, "filters.and.filters:"),
    (
        "untyped-member",
        {"type": "and", "filters": [{"type": "eq", "key": "department", "value": "hr"}, {"key": "year"}]},
        "filters.and.filters.1",
    ),
    ("bools-in-list", {"type": "in", "key": "remote", "value": [True, False]}, "filters.in.value"),
    ("not-an-object", [{"type": "eq", "key": "department", "value": "hr"}], "filters must be an object, got list"),
    ("empty-object", {}, "filters must not be empty"),
    (
        "extra-key-on-comparison",
        {"type": "eq", "key": "department", "value": "hr", "filters": [{"type": "eq", "key": "year", "value": 2024}]},
        "filters.eq.filters: Extra inputs are not permitted",
    ),
    (
        "extra-key-on-compound",
        {"type": "or", "filters": [{"type": "eq", "key": "department", "value": "hr"}], "key": "year"},
        "filters.or.key: Extra inputs are not permitted",
    ),
    ("in-scalar", {"type": "in", "key": "department", "value": "hr"}, "filters.in.value"),
    ("in-empty-list", {"type": "in", "key": "department", "value": []}, "filters.in.value: Value error, must not be"),
    ("eq-list", {"type": "eq", "key": "department", "value": ["hr"]}, "filters.eq.value"),
    ("gt-string", {"type": "gt", "key": "year", "value": "2023"}, "filters.gt.value"),
    ("gte-bool", {"type": "gte", "key": "year", "value": True}, "filters.gte.value"),
)


@pytest.mark.parametrize(
    ("filters", "reason"), [pytest.param(filters, reason, id=case) for case, filters, reason in INVALID_FILTERS]
)
@pytest.mark.asyncio
async def test_search_rejects_invalid_filters_with_400_without_calling_aws(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, filters: object, reason: str
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", False)
    retrieve_route: Final = respx_mock.post(BEDROCK_RETRIEVE_URL).mock(
        return_value=httpx.Response(200, json=BEDROCK_RETRIEVE_RESPONSE)
    )

    with pytest.raises(litellm.BadRequestError) as error:
        await litellm.vector_stores.asearch(
            vector_store_id="T37J8R4WTM",
            query="vacation policy",
            filters=filters,
            custom_llm_provider="bedrock",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            aws_region_name="us-east-1",
        )

    assert error.value.status_code == 400
    assert reason in error.value.message
    assert retrieve_route.call_count == 0


@pytest.mark.parametrize("filters", [pytest.param(filters, id=case) for case, filters, _ in INVALID_FILTERS])
@pytest.mark.parametrize("drop_params_source", ["request", "global"])
@pytest.mark.asyncio
async def test_search_drops_invalid_filters_under_drop_params(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, filters: object, drop_params_source: str
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", drop_params_source == "global")
    request_kwargs: Final = {"drop_params": True} if drop_params_source == "request" else {}

    retrieve_route: Final = await _search_with_filters(respx_mock, filters, **request_kwargs)

    assert retrieve_route.call_count == 1
    assert _sent_filter(retrieve_route) is None


async def _rejection(respx_mock: respx.MockRouter, filters: object) -> BaseException:
    import litellm

    with pytest.raises(litellm.BadRequestError) as error:
        await _search_with_filters(respx_mock, filters)
    return error.value


def _nested_compound(depth: int) -> Mapping[str, object]:
    if depth == 0:
        return {"type": "eq", "key": "year"}
    return {"type": "and", "filters": [_nested_compound(depth - 1)]}


@pytest.mark.parametrize(
    ("one_bad_item", "many_bad_items"),
    [
        pytest.param(
            {"type": "in", "key": "remote", "value": [True]},
            {"type": "in", "key": "remote", "value": [True] * 1000},
            id="bools-in-list",
        ),
        pytest.param(
            {"type": "and", "filters": [{"key": "year"}]},
            {"type": "and", "filters": [{"key": "year"}] * 1000},
            id="untyped-members",
        ),
    ],
)
@pytest.mark.asyncio
async def test_search_rejection_names_the_first_bad_item_however_many_follow(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    one_bad_item: dict[str, object],
    many_bad_items: dict[str, object],
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", False)

    assert str(await _rejection(respx_mock, many_bad_items)) == str(await _rejection(respx_mock, one_bad_item))


@pytest.mark.asyncio
async def test_search_rejection_and_its_traceback_do_not_grow_with_nesting_depth(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", False)

    shallow: Final = await _rejection(respx_mock, _nested_compound(160))
    deep: Final = await _rejection(respx_mock, _nested_compound(240))

    assert len(str(deep)) == len(str(shallow))
    assert "eq.value" in str(deep)
    assert len("".join(traceback.format_exception(deep))) == len("".join(traceback.format_exception(shallow)))


@pytest.mark.asyncio
async def test_search_rejection_names_a_bad_member_the_same_wherever_it_sits(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", False)
    good: Final = {"type": "eq", "key": "department", "value": "hr"}
    bad: Final = {"key": "year"}

    bad_first: Final = str(await _rejection(respx_mock, {"type": "and", "filters": [bad, good]}))
    bad_last: Final = str(await _rejection(respx_mock, {"type": "and", "filters": [good, bad]}))

    assert bad_first.replace("filters.and.filters.0", "filters.and.filters.1") == bad_last
    assert "..." not in bad_last


def _image_search_body(media_type: str, extra_body: dict[str, object] | None) -> dict[str, object]:
    config: Final = BedrockVectorStoreConfig()
    mock_log: Final = MagicMock()
    mock_log.model_call_details = {}
    _, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="",
        vector_store_search_optional_params={
            "max_num_results": 3,
            "query_image": {"type": "base64", "media_type": media_type, "data": "aW1hZ2UtYnl0ZXM="},
        },
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
        extra_body=extra_body,
    )
    return body


@pytest.mark.parametrize(
    ("media_type", "bedrock_format"),
    [("image/jpeg", "jpeg"), ("image/png", "png"), ("image/gif", "gif"), ("image/webp", "webp")],
)
def test_transform_search_request_sends_query_image_as_an_image_retrieval_query(
    media_type: str, bedrock_format: str
) -> None:
    body: Final = _image_search_body(media_type, extra_body=None)

    assert body["retrievalQuery"] == {
        "type": "IMAGE",
        "image": {"format": bedrock_format, "inlineContent": "aW1hZ2UtYnl0ZXM="},
    }
    assert body["retrievalConfiguration"] == {"vectorSearchConfiguration": {"numberOfResults": 3}}


CALLER_RETRIEVAL_QUERY: Final = {"type": "IMAGE", "image": {"format": "png", "inlineContent": "Y2FsbGVy"}}


@pytest.mark.parametrize("key", ["retrievalQuery", "retrieval_query"])
@pytest.mark.parametrize("source", ["extra_body", "litellm_params"])
def test_transform_search_request_rejects_a_callers_retrieval_query(key: str, source: str) -> None:
    caller_fields: Final = {key: CALLER_RETRIEVAL_QUERY}

    with pytest.raises(litellm.BadRequestError) as exc_info:
        _search_body(
            extra_body=caller_fields if source == "extra_body" else None,
            litellm_params=caller_fields if source == "litellm_params" else {},
        )

    assert exc_info.value.status_code == 400
    assert "query_image" in str(exc_info.value)


def test_transform_search_request_drops_a_callers_retrieval_query_when_drop_params_is_on() -> None:
    body: Final = _search_body(
        extra_body={"retrievalQuery": CALLER_RETRIEVAL_QUERY}, litellm_params={"drop_params": True}
    )

    assert body["retrievalQuery"] == {"text": "hello"}


QUERY_IMAGE: Final = {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="}
IMAGE_HIT_METADATA: Final = {
    "x-amz-bedrock-kb-source-uri": "s3://kb-images/shapes/red-circle.jpg",
    "x-amz-bedrock-kb-source-file-mime-type": "image/jpeg",
    "x-amz-bedrock-kb-source-file-modality": "IMAGE",
}
TEXT_HIT_METADATA: Final = {"x-amz-bedrock-kb-source-uri": "s3://kb-docs/shapes.txt"}
MIXED_RETRIEVAL_RESULTS: Final = [
    {
        "content": {"type": "IMAGE", "byteContent": "data:image/png;base64,/9j/4AAQSkZJRg=="},
        "metadata": IMAGE_HIT_METADATA,
        "location": {"type": "S3", "s3Location": {"uri": "s3://kb-images/shapes/red-circle.jpg"}},
        "score": 0.998,
    },
    {"content": {"type": "TEXT", "text": "a red circle"}, "metadata": TEXT_HIT_METADATA, "score": 0.5},
    {"content": {"type": "ROW", "row": []}, "metadata": {}, "score": 0.4},
    {"content": {"type": "AUDIO", "byteContent": "data:audio/mpeg;base64,SUQz"}, "metadata": {}, "score": 0.3},
]
TEXT_HIT: Final = {
    "score": 0.5,
    "content": [{"text": "a red circle", "type": "text"}],
    "file_id": "s3://kb-docs/shapes.txt",
    "filename": "shapes.txt",
    "attributes": TEXT_HIT_METADATA,
}


def _search_response(
    retrieval_results: list[dict[str, object]], query: str = "", query_image: dict[str, str] | None = QUERY_IMAGE
) -> dict[str, object]:
    config: Final = BedrockVectorStoreConfig()
    mock_log: Final = MagicMock()
    mock_log.model_call_details = {"query": query, "query_image": query_image}
    return config.transform_search_vector_store_response(
        response=httpx.Response(200, json={"retrievalResults": retrieval_results}),
        litellm_logging_obj=mock_log,
    )


def test_transform_search_response_to_an_image_search_returns_image_hits_next_to_text_hits() -> None:
    response: Final = _search_response(MIXED_RETRIEVAL_RESULTS)

    assert response["search_query"] == ""
    assert response["data"] == [
        {
            "score": 0.998,
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="},
                }
            ],
            "file_id": "s3://kb-images/shapes/red-circle.jpg",
            "filename": "red-circle.jpg",
            "attributes": IMAGE_HIT_METADATA,
        },
        TEXT_HIT,
    ]


def test_transform_search_response_to_a_text_search_returns_only_text_hits() -> None:
    response: Final = _search_response(MIXED_RETRIEVAL_RESULTS, query="red circle", query_image=None)

    assert response["search_query"] == "red circle"
    assert response["data"] == [TEXT_HIT]


@pytest.mark.parametrize(
    ("byte_content", "metadata", "media_type", "data"),
    [
        ("data:image/webp;base64,UklGRg==", {}, "image/webp", "UklGRg=="),
        ("data:image/png;base64,iVBORw0K", {"x-amz-bedrock-kb-source-file-mime-type": ""}, "image/png", "iVBORw0K"),
        (
            "data:image/png;base64,iVBORw0K",
            {"x-amz-bedrock-kb-source-file-mime-type": "application/pdf"},
            "image/png",
            "iVBORw0K",
        ),
        ("/9j/4AAQ", {"x-amz-bedrock-kb-source-file-mime-type": "image/jpeg"}, "image/jpeg", "/9j/4AAQ"),
        ("/9j/4AAQ", {}, "application/octet-stream", "/9j/4AAQ"),
    ],
)
def test_transform_search_response_image_hit_media_type_falls_back_to_the_data_uri(
    byte_content: str, metadata: dict[str, object], media_type: str, data: str
) -> None:
    response: Final = _search_response(
        [{"content": {"type": "IMAGE", "byteContent": byte_content}, "metadata": metadata, "score": 0.9}]
    )

    assert response["data"][0]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    ]


class _SearchLogging:
    def __init__(self, query: str) -> None:
        self.model_call_details: Final = {"query": query}


def test_search_response_keeps_each_text_result_with_its_score_and_metadata() -> None:
    metadata: Final = {"x-amz-bedrock-kb-source-uri": "s3://docs/setup.md", "x-amz-bedrock-kb-chunk-id": "chunk-1"}
    body: Final = {
        "retrievalResults": [
            {"content": {"text": "Run litellm --config config.yaml"}, "metadata": metadata, "score": 0.42},
            {"content": {"type": "IMAGE"}, "score": 0.9},
        ]
    }

    response: Final = BedrockVectorStoreConfig().transform_search_vector_store_response(
        httpx.Response(200, json=body), _SearchLogging("how do I start the proxy")
    )

    assert response["search_query"] == "how do I start the proxy"
    assert [(result["score"], result["content"][0]["text"], result["attributes"]) for result in response["data"]] == [
        (0.42, "Run litellm --config config.yaml", metadata)
    ]


@pytest.mark.parametrize("body", [[], [{"retrievalResults": []}], "retrievalResults"])
def test_search_response_that_is_not_a_json_object_raises_a_bedrock_error(body: object) -> None:
    with pytest.raises(BedrockError) as raised:
        BedrockVectorStoreConfig().transform_search_vector_store_response(
            httpx.Response(200, json=body), _SearchLogging("query")
        )

    assert raised.value.status_code == 200
