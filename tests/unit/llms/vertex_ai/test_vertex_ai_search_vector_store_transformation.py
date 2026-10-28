import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final

import httpx
import pytest
import respx

from litellm.exceptions import BadRequestError
from litellm.llms.vertex_ai.vector_stores.search_api.transformation import (
    VertexSearchAPIVectorStoreConfig,
)
from litellm.types.vector_stores import VectorStoreSearchResponse


def test_should_encode_vertex_search_vector_store_id_in_complete_url():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_collection_id": "default/collection",
            "vector_store_id": "../../dataStores/other?x=1#frag",
        },
    )

    assert (
        url
        == "https://discoveryengine.googleapis.com/v1/projects/test-project/locations/global/collections/default%2Fcollection/dataStores/..%2F..%2FdataStores%2Fother%3Fx%3D1%23frag/servingConfigs/default_config"
    )


def test_should_reject_dot_segment_vertex_search_vector_store_id():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(
        ValueError, match="vector_store_id cannot be a dot path segment"
    ):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
                "vector_store_id": "..",
            },
        )


def test_should_use_engines_url_when_engine_id_provided():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "test-engine_1234",
        },
    )

    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/test-project/locations/global/"
        "collections/default_collection/engines/test-engine_1234/servingConfigs/default_serving_config"
    )


def test_engine_id_takes_precedence_over_vector_store_id():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "test-engine_1234",
            "vector_store_id": "ignored-when-engine-set",
        },
    )

    assert "/engines/test-engine_1234/" in url
    assert "/dataStores/" not in url
    assert url.endswith("/servingConfigs/default_serving_config")


def test_should_encode_vertex_engine_id_in_complete_url():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "../../engines/other?x=1#frag",
        },
    )

    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/test-project/locations/global/"
        "collections/default_collection/engines/..%2F..%2Fengines%2Fother%3Fx%3D1%23frag/servingConfigs/default_serving_config"
    )


def test_should_reject_dot_segment_vertex_engine_id():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(
        ValueError, match="vertex_engine_id cannot be a dot path segment"
    ):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
                "vertex_engine_id": "..",
            },
        )


def test_should_raise_when_neither_engine_id_nor_vector_store_id_provided():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(
        ValueError,
        match="vector_store_id is required when vertex_engine_id is not set",
    ):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
        )


_ENGINE_BASE = (
    "https://discoveryengine.googleapis.com/v1/projects/p/locations/global/"
    "collections/default_collection/engines/app-2/servingConfigs/default_serving_config"
)

_DATASTORE_BASE = (
    "https://discoveryengine.googleapis.com/v1/projects/p/locations/global/"
    "collections/default_collection/dataStores/ds-1/servingConfigs/default_config"
)


def _search_request(**overrides):
    """Engine/app-mode search request (vertex_engine_id set)."""
    kwargs = dict(
        vector_store_id="vs",
        query="hello",
        vector_store_search_optional_params={},
        api_base=_ENGINE_BASE,
        litellm_logging_obj=SimpleNamespace(model_call_details={}),
        litellm_params={"vertex_engine_id": "app-2"},
    )
    kwargs.update(overrides)
    return VertexSearchAPIVectorStoreConfig().transform_search_vector_store_request(
        **kwargs
    )


def _datastore_search_request(**overrides):
    """Data-store-mode search request (no vertex_engine_id)."""
    kwargs = dict(
        vector_store_id="ds-1",
        query="hello",
        vector_store_search_optional_params={},
        api_base=_DATASTORE_BASE,
        litellm_logging_obj=SimpleNamespace(model_call_details={}),
        litellm_params={},
    )
    kwargs.update(overrides)
    return VertexSearchAPIVectorStoreConfig().transform_search_vector_store_request(
        **kwargs
    )


def test_search_request_defaults_to_query_and_pagesize_10():
    url, body = _search_request()

    assert url == _ENGINE_BASE + ":search"
    assert body == {"query": "hello", "pageSize": 10}


def test_search_request_maps_max_num_results_to_pagesize():
    _, body = _search_request(
        vector_store_search_optional_params={"max_num_results": 25}
    )

    assert body["pageSize"] == 25


def test_engine_search_request_forwards_datastorespecs():
    specs = [
        {
            "dataStore": "projects/p/locations/global/collections/default_collection/dataStores/ds-beta"
        }
    ]

    _, body = _search_request(extra_body={"dataStoreSpecs": specs})

    assert body["dataStoreSpecs"] == specs


def test_engine_search_request_forwards_num_results_per_data_store():
    _, body = _search_request(extra_body={"numResultsPerDataStore": 3})

    assert body["numResultsPerDataStore"] == 3


def test_datastore_search_request_rejects_datastorespecs():
    specs = [{"dataStore": "projects/p/.../dataStores/ds-beta"}]

    with pytest.raises(BadRequestError, match="data store mode"):
        _datastore_search_request(extra_body={"dataStoreSpecs": specs})


def test_datastore_search_request_rejects_num_results_per_data_store():
    with pytest.raises(BadRequestError, match="data store mode"):
        _datastore_search_request(extra_body={"numResultsPerDataStore": 3})


@pytest.mark.parametrize("field", ["branch", "servingConfig", "entity"])
def test_search_request_rejects_target_selecting_fields(field):
    with pytest.raises(BadRequestError, match="target-selecting"):
        _search_request(extra_body={field: "x"})


@pytest.mark.parametrize("field", ["branch", "servingConfig", "entity"])
def test_datastore_search_request_rejects_target_selecting_fields(field):
    with pytest.raises(BadRequestError, match="target-selecting"):
        _datastore_search_request(extra_body={field: "x"})


def test_search_request_rejects_unsupported_extra_body_field():
    with pytest.raises(BadRequestError, match="Unsupported Vertex AI Search extra_body"):
        _search_request(extra_body={"notARealField": True})


def test_rejected_extra_body_raises_http_400():
    with pytest.raises(BadRequestError) as exc_info:
        _search_request(extra_body={"notARealField": True})

    assert exc_info.value.status_code == 400


def test_search_request_forwards_supported_extra_body_fields():
    _, body = _search_request(
        extra_body={
            "filter": 'category: ANY("docs")',
            "boostSpec": {"conditionBoostSpecs": []},
        }
    )

    assert body["filter"] == 'category: ANY("docs")'
    assert body["boostSpec"] == {"conditionBoostSpecs": []}
    assert body["query"] == "hello"


def test_datastore_search_request_forwards_supported_extra_body_fields():
    _, body = _datastore_search_request(
        extra_body={"filter": 'category: ANY("docs")'}
    )

    assert body["filter"] == 'category: ANY("docs")'


def test_search_request_ignores_none_valued_extra_body_fields():
    _, body = _search_request(extra_body={"filter": None})

    assert "filter" not in body


def test_search_request_extra_body_takes_precedence_over_defaults():
    _, body = _search_request(
        vector_store_search_optional_params={"max_num_results": 5},
        extra_body={"pageSize": 50, "filter": 'category: ANY("docs")'},
    )

    assert body["pageSize"] == 50
    assert body["filter"] == 'category: ANY("docs")'


def test_search_request_joins_list_query():
    _, body = _search_request(query=["foo", "bar"])

    assert body["query"] == "foo bar"


def test_search_request_logs_effective_query_when_extra_body_overrides_query():
    log = SimpleNamespace(model_call_details={})

    _, body = _search_request(
        query="original",
        extra_body={"query": "from-extra-body"},
        litellm_logging_obj=log,
    )

    assert body["query"] == "from-extra-body"
    assert log.model_call_details["query"] == "from-extra-body"


_CHUNK_NAME = (
    "projects/p/locations/global/collections/default_collection/dataStores/ds-1/"
    "branches/0/documents/policy/chunks/c3"
)


def _search_response(payload: Mapping[str, object]) -> VectorStoreSearchResponse:
    return VertexSearchAPIVectorStoreConfig().transform_search_vector_store_response(
        response=SimpleNamespace(json=lambda: payload, status_code=200, headers={}),
        litellm_logging_obj=SimpleNamespace(model_call_details={"query": "hello"}),
    )


def test_chunk_hit_uses_chunk_content_and_document_metadata():
    payload = {
        "results": [
            {
                "chunk": {
                    "id": "c3",
                    "name": _CHUNK_NAME,
                    "content": "Refunds are available within 14 days.",
                    "documentMetadata": {
                        "uri": "gs://bucket/policy.pdf",
                        "title": "Refund policy",
                        "structData": {"department": "billing"},
                    },
                    "pageSpan": {"pageStart": 2, "pageEnd": 2},
                    "relevanceScore": 0.91,
                }
            }
        ]
    }

    result = _search_response(payload)["data"][0]

    assert result["content"] == [
        {"text": "Refunds are available within 14 days.", "type": "text"}
    ]
    assert result["score"] == 0.91
    assert result["file_id"] == "gs://bucket/policy.pdf"
    assert result["filename"] == "Refund policy"
    assert result["attributes"] == {
        "document_id": "policy",
        "chunk_id": "c3",
        "link": "gs://bucket/policy.pdf",
        "title": "Refund policy",
        "structData": {"department": "billing"},
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
    }


def test_chunk_hit_without_uri_or_title_falls_back_to_document_id():
    payload = {
        "results": [
            {
                "chunk": {
                    "id": "c1",
                    "name": _CHUNK_NAME.replace("policy/chunks/c3", "handbook/chunks/c1"),
                    "content": "Guest Services Handbook",
                    "documentMetadata": {"structData": {"title": "Handbook"}},
                }
            }
        ]
    }

    result = _search_response(payload)["data"][0]

    assert result["content"] == [{"text": "Guest Services Handbook", "type": "text"}]
    assert result["score"] == 1.0
    assert result["file_id"] == "handbook"
    assert result["filename"] == "Unknown Document"
    assert result["attributes"] == {
        "document_id": "handbook",
        "chunk_id": "c1",
        "structData": {"title": "Handbook"},
    }


@pytest.mark.parametrize(
    ("derived", "expected_text"),
    [
        (
            {
                "extractive_segments": [{"content": "seg one"}, {"content": "seg two"}],
                "extractive_answers": [{"content": "ans"}],
                "snippets": [{"snippet": "snip"}],
                "title": "policy.pdf",
            },
            "seg one\n\nseg two",
        ),
        (
            {
                "extractive_answers": [{"content": "ans one"}, {"content": "ans two"}],
                "snippets": [{"snippet": "snip"}],
                "title": "policy.pdf",
            },
            "ans one\n\nans two",
        ),
        (
            {
                "snippets": [{"snippet": "snip a"}, {"htmlSnippet": "snip b"}],
                "title": "policy.pdf",
            },
            "snip a snip b",
        ),
        (
            {"extractive_segments": [{"pageNumber": "1"}, {"content": "seg", "pageNumber": "2"}]},
            "seg",
        ),
        ({"title": "policy.pdf"}, "policy.pdf"),
    ],
    ids=["segments", "answers", "snippets", "content_less_segment", "title"],
)
def test_document_hit_text_prefers_extractive_content(derived: Mapping[str, object], expected_text: str) -> None:
    payload = {"results": [{"id": "doc-1", "document": {"derivedStructData": derived}}]}

    result = _search_response(payload)["data"][0]

    assert result["content"] == [{"text": expected_text, "type": "text"}]


def test_document_hit_surfaces_struct_data_in_attributes():
    payload = {
        "results": [
            {
                "id": "attr-1",
                "document": {
                    "structData": {"title": "Thunder Loop", "waitMinutes": 45},
                    "derivedStructData": {"clearbox_escorer_score": 0.5},
                },
            },
            {"id": "attr-2", "document": {"structData": {}, "derivedStructData": {}}},
        ]
    }

    first, second = _search_response(payload)["data"]

    assert first["content"] == [{"text": "", "type": "text"}]
    assert first["file_id"] == "attr-1"
    assert first["filename"] == "Unknown Document"
    assert first["attributes"] == {
        "document_id": "attr-1",
        "structData": {"title": "Thunder Loop", "waitMinutes": 45},
    }
    assert second["attributes"] == {"document_id": "attr-2"}


def test_search_response_keeps_link_metadata_and_positional_scores():
    payload = {
        "results": [
            {
                "id": "doc-1",
                "document": {
                    "derivedStructData": {
                        "title": "Terms",
                        "link": "gs://bucket/terms.pdf",
                        "displayLink": "bucket",
                        "formattedUrl": "https://bucket/terms.pdf",
                        "snippets": [{"snippet": "snip"}],
                    }
                },
            },
            {"chunk": {"name": _CHUNK_NAME, "content": "chunk text"}},
        ]
    }

    response = _search_response(payload)

    assert response["object"] == "vector_store.search_results.page"
    assert response["search_query"] == "hello"
    assert [result["score"] for result in response["data"]] == [1.0, 0.5]
    assert response["data"][0]["file_id"] == "gs://bucket/terms.pdf"
    assert response["data"][0]["attributes"] == {
        "document_id": "doc-1",
        "link": "gs://bucket/terms.pdf",
        "title": "Terms",
        "displayLink": "bucket",
        "formattedUrl": "https://bucket/terms.pdf",
    }


def test_search_response_without_results_key_is_empty():
    assert _search_response({"totalSize": 0})["data"] == []


MOCK_VERTEX_SEARCH_RESPONSE: Final = {
    "results": [
        {
            "id": "0",
            "document": {
                "name": "projects/648660250433/locations/global/collections/default_collection/dataStores/litellm-docs_1761094140318/branches/0/documents/0",
                "id": "0",
                "derivedStructData": {
                    "htmlTitle": "<b>LiteLLM</b> - Getting Started | <b>liteLLM</b>",
                    "snippets": [
                        {
                            "htmlSnippet": "https://github.com/BerriAI/<b>litellm</b>.",
                            "snippet": "https://github.com/BerriAI/litellm.",
                        }
                    ],
                    "title": "LiteLLM - Getting Started | liteLLM",
                    "link": "https://docs.litellm.ai/docs/",
                    "displayLink": "docs.litellm.ai",
                },
            },
        },
        {
            "id": "1",
            "document": {
                "name": "projects/648660250433/locations/global/collections/default_collection/dataStores/litellm-docs_1761094140318/branches/0/documents/1",
                "id": "1",
                "derivedStructData": {
                    "title": "Using Vector Stores (Knowledge Bases) | liteLLM",
                    "link": "https://docs.litellm.ai/docs/completion/knowledgebase",
                    "snippets": [
                        {
                            "snippet": "LiteLLM integrates with vector stores, allowing your models to access your organization's data."
                        }
                    ],
                },
            },
        },
    ],
    "totalSize": 299,
    "attributionToken": "mock_token",
    "summary": {},
}
GOOGLE_TOKEN_URL: Final = "https://oauth2.googleapis.com/token"
VERTEX_AUTHORIZED_USER_CREDENTIALS: Final = json.dumps(
    {
        "type": "authorized_user",
        "client_id": "unit-client-id",
        "client_secret": "unit-client-secret",
        "refresh_token": "unit-refresh-token",
        "quota_project_id": "test-vector-store-db",
    }
)


def _mock_google_token_refresh(requests_mock) -> None:
    requests_mock.post(
        GOOGLE_TOKEN_URL,
        json={"access_token": "ya29.unit-token", "expires_in": 3600, "token_type": "Bearer"},
    )


def _mock_vertex_search(respx_mock) -> "respx.Route":
    return respx_mock.post(url__regex=r"https://discoveryengine\.googleapis\.com/.*:search.*").mock(
        return_value=httpx.Response(200, json=MOCK_VERTEX_SEARCH_RESPONSE)
    )


@pytest.mark.asyncio
async def test_basic_search_with_mock(respx_mock, requests_mock, monkeypatch: pytest.MonkeyPatch):
    """Test basic vector search with mocked backend response"""
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _mock_google_token_refresh(requests_mock)
    search_route = _mock_vertex_search(respx_mock)

    response = await litellm.vector_stores.asearch(
        query="what is LiteLLM?",
        vector_store_id="test-litellm-app_1761094730750",
        custom_llm_provider="vertex_ai/search_api",
        vertex_project="test-vector-store-db",
        vertex_location="us-central1",
        vertex_credentials=VERTEX_AUTHORIZED_USER_CREDENTIALS,
    )

    assert response is not None
    assert response["object"] == "vector_store.search_results.page"
    assert len(response["data"]) > 0
    assert "search_query" in response

    first_result = response["data"][0]
    assert len(first_result["content"]) > 0
    assert first_result["content"][0]["type"] == "text"
    assert "text" in first_result["content"][0]

    search_request = search_route.calls.last.request
    url = str(search_request.url)
    assert "discoveryengine.googleapis.com" in url
    assert "test-vector-store-db" in url
    assert "test-litellm-app_1761094730750" in url
    assert search_request.headers["authorization"] == "Bearer ya29.unit-token"


def test_basic_search_sync_with_mock(respx_mock, requests_mock):
    """Test basic vector search (sync) with mocked backend response"""
    _mock_google_token_refresh(requests_mock)
    search_route = _mock_vertex_search(respx_mock)

    response = __import__("litellm").vector_stores.search(
        query="what is LiteLLM?",
        vector_store_id="test-litellm-app_1761094730750",
        custom_llm_provider="vertex_ai/search_api",
        vertex_project="test-vector-store-db",
        vertex_location="us-central1",
        vertex_credentials=VERTEX_AUTHORIZED_USER_CREDENTIALS,
    )

    assert response is not None
    assert response["object"] == "vector_store.search_results.page"
    assert len(response["data"]) > 0
    assert "search_query" in response

    first_result = response["data"][0]
    assert len(first_result["content"]) > 0
    assert first_result["content"][0]["type"] == "text"
    assert "text" in first_result["content"][0]

    assert "document_id" in first_result["attributes"]
    assert "link" in first_result["attributes"]
    assert "title" in first_result["attributes"]

    assert search_route.call_count == 1
    assert search_route.calls.last.request.headers["authorization"] == "Bearer ya29.unit-token"
