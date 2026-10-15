import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.qdrant.vector_stores.transformation import QdrantVectorStoreConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import VectorStoreSearchOptionalRequestParams

_VECTOR: Final = [0.5, -0.25, 0.125]
_BASE: Final = "https://qdrant.example"
_URL: Final = f"{_BASE}/collections/documents/points/query"
_EMBEDDING_URL: Final = "https://api.openai.com/v1/embeddings"


@dataclass(frozen=True, slots=True)
class Embeddings:
    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return EmbeddingResponse(data=[{"object": "embedding", "index": 0, "embedding": _VECTOR}])

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.embed(model, query, configuration)


def _logging(params: Mapping[str, object] | None = None) -> Logging:
    logging: Final = Logging(
        model="documents",
        messages="question",
        stream=False,
        call_type="vector_store_search",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="qdrant-test",
        function_id="qdrant-test",
    )
    logging.model_call_details["litellm_params"] = dict(params or {})
    logging.model_call_details["qdrant_search_query"] = "question"
    return logging


def _upstream_routes(mock: respx.MockRouter) -> tuple[respx.Route, respx.Route]:
    embedding: Final = mock.post(_EMBEDDING_URL).respond(
        200,
        json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": _VECTOR}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )
    search: Final = mock.post(_URL).respond(
        200,
        json={
            "result": {
                "points": [{"id": 42, "score": 0.9, "payload": {"body": "Retrieved document", "category": "research"}}]
            }
        },
    )
    return embedding, search


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_sdk_search_embeds_query_and_returns_payload_text(
    respx_mock: respx.MockRouter, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    embedding, search = _upstream_routes(respx_mock)
    kwargs: Final = {
        "vector_store_id": "documents",
        "query": "Find research",
        "custom_llm_provider": "qdrant",
        "api_base": _BASE,
        "api_key": "qdrant-test-key",
        "litellm_embedding_model": "openai/text-embedding-3-small",
        "litellm_embedding_config": {"api_key": "embedding-test-key"},
        "qdrant_text_field": "body",
        "qdrant_vector_name": "dense",
        "max_num_results": 3,
        "filters": {"type": "eq", "key": "category", "value": "research"},
        "ranking_options": {"score_threshold": 0.8},
    }
    response: Final = (
        await litellm.vector_stores.asearch(**kwargs) if asynchronous else litellm.vector_stores.search(**kwargs)
    )
    assert json.loads(embedding.calls.last.request.content) == {
        "model": "text-embedding-3-small",
        "input": ["Find research"],
    }
    assert embedding.calls.last.request.headers["authorization"] == "Bearer embedding-test-key"
    assert search.calls.last.request.headers["api-key"] == "qdrant-test-key"
    assert json.loads(search.calls.last.request.content) == {
        "query": _VECTOR,
        "using": "dense",
        "limit": 3,
        "with_payload": True,
        "with_vector": False,
        "filter": {"must": [{"key": "category", "match": {"value": "research"}}]},
        "score_threshold": 0.8,
    }
    assert response == {
        "object": "vector_store.search_results.page",
        "search_query": "Find research",
        "data": [
            {
                "score": 0.9,
                "content": [{"type": "text", "text": "Retrieved document"}],
                "file_id": "42",
                "filename": None,
                "attributes": {"category": "research"},
            }
        ],
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_router_search_resolves_proxy_embedding_alias(
    respx_mock: respx.MockRouter, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedding, search = _upstream_routes(respx_mock)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "query-embeddings",
                "litellm_params": {"model": "openai/text-embedding-3-small", "api_key": "router-key"},
            }
        ]
    )
    kwargs: Final = {
        "vector_store_id": "documents",
        "query": "Find research",
        "custom_llm_provider": "qdrant",
        "api_base": _BASE,
        "api_key": "store-key",
        "qdrant_text_field": "body",
        "litellm_embedding_model": "query-embeddings",
    }
    response: Final = (
        await router.avector_store_search(**kwargs) if asynchronous else router.vector_store_search(**kwargs)
    )
    assert embedding.calls.last.request.headers["authorization"] == "Bearer router-key"
    assert json.loads(search.calls.last.request.content)["query"] == _VECTOR
    assert response["data"] == [
        {
            "score": 0.9,
            "content": [{"type": "text", "text": "Retrieved document"}],
            "file_id": "42",
            "filename": None,
            "attributes": {"category": "research"},
        }
    ]


def test_default_vector_and_native_filter_preserve_collection_scope() -> None:
    url, body = QdrantVectorStoreConfig().transform_search_vector_store_request(
        vector_store_id="folder/documents",
        query=["find", "research"],
        vector_store_search_optional_params={},
        api_base=_BASE,
        litellm_logging_obj=_logging(),
        litellm_params={"litellm_embedding_model": "query-embeddings", "qdrant_text_field": None},
        extra_body={"filter": {"must": [{"key": "language", "match": {"value": "fr"}}]}, "params": {"exact": True}},
        embedding_executor=Embeddings(),
    )
    assert url == f"{_BASE}/collections/folder%2Fdocuments/points/query"
    assert body == {
        "query": _VECTOR,
        "limit": 10,
        "with_payload": True,
        "with_vector": False,
        "filter": {"must": [{"key": "language", "match": {"value": "fr"}}]},
        "params": {"exact": True},
    }


@pytest.mark.parametrize(
    "text_field,payload",
    [
        (None, {"text": "Document"}),
        ("metadata.body", {"metadata": {"body": "Document"}}),
        ("metadata.body", {"metadata.body": "Document"}),
    ],
)
def test_response_reads_default_and_nested_text_fields(text_field: str | None, payload: Mapping[str, object]) -> None:
    response: Final = QdrantVectorStoreConfig().transform_search_vector_store_response(
        httpx.Response(200, json={"result": {"points": [{"id": "uuid", "score": 0.25, "payload": payload}]}}),
        _logging({"qdrant_text_field": text_field}),
    )
    assert response["data"][0]["content"] == [{"type": "text", "text": "Document"}]
    assert response["data"][0]["file_id"] == "uuid"


def test_response_preserves_source_file_metadata_and_raw_score() -> None:
    response: Final = QdrantVectorStoreConfig().transform_search_vector_store_response(
        httpx.Response(
            200,
            json={
                "result": {
                    "points": [
                        {
                            "id": 1,
                            "score": 12.5,
                            "payload": {"text": "Document", "file_id": "source-1", "filename": "paper.pdf", "page": 2},
                        }
                    ]
                }
            },
        ),
        _logging(),
    )
    assert response["data"] == [
        {
            "score": 12.5,
            "content": [{"type": "text", "text": "Document"}],
            "file_id": "source-1",
            "filename": "paper.pdf",
            "attributes": {"file_id": "source-1", "filename": "paper.pdf", "page": 2},
        }
    ]


def test_empty_search_returns_no_documents() -> None:
    response: Final = QdrantVectorStoreConfig().transform_search_vector_store_response(
        httpx.Response(200, json={"result": {"points": []}}),
        _logging(),
    )
    assert response == {"object": "vector_store.search_results.page", "search_query": "question", "data": []}


@pytest.mark.parametrize("payload", [None, {}, {"text": ["not text"]}, {"text": {"body": "not a string"}}])
def test_missing_or_invalid_text_fails_instead_of_returning_empty_context(payload: object) -> None:
    with pytest.raises(litellm.BadRequestError, match="must contain text"):
        QdrantVectorStoreConfig().transform_search_vector_store_response(
            httpx.Response(200, json={"result": {"points": [{"id": 1, "score": 0.9, "payload": payload}]}}),
            _logging(),
        )


@pytest.mark.parametrize(
    "options,extra",
    [
        ({"max_num_results": 0}, None),
        ({"max_num_results": 51}, None),
        ({"max_num_results": True}, None),
        ({"rewrite_query": True}, None),
        ({"ranking_options": {"ranker": "custom"}}, None),
        ({"ranking_options": {"score_threshold": -0.1}}, None),
        ({"filters": {"type": "eq", "key": "language", "value": "fr"}}, {"filter": {}}),
        ({"ranking_options": {"score_threshold": 0.5}}, {"score_threshold": 0.5}),
        ({}, {"query": [1, 0, 0]}),
        ({}, {"using": "another-vector"}),
        ({}, {"with_payload": False}),
        ({}, {"score_threshold": float("nan")}),
    ],
)
def test_unsupported_options_fail_before_embedding(
    options: VectorStoreSearchOptionalRequestParams,
    extra: Mapping[str, object] | None,
) -> None:
    with pytest.raises(litellm.BadRequestError):
        QdrantVectorStoreConfig().transform_search_vector_store_request(
            vector_store_id="documents",
            query="q",
            vector_store_search_optional_params=options,
            api_base=_BASE,
            litellm_logging_obj=_logging(),
            litellm_params={},
            extra_body=extra,
        )


def test_requires_embedding_model() -> None:
    with pytest.raises(ValueError, match="litellm_embedding_model is required"):
        QdrantVectorStoreConfig().transform_search_vector_store_request(
            vector_store_id="documents",
            query="q",
            vector_store_search_optional_params={},
            api_base=_BASE,
            litellm_logging_obj=_logging(),
            litellm_params={},
            embedding_executor=Embeddings(),
        )


@pytest.mark.parametrize("query", ["", " ", []])
def test_empty_query_is_rejected(query: str | Sequence[str]) -> None:
    with pytest.raises(litellm.BadRequestError, match="query must not be empty"):
        QdrantVectorStoreConfig().transform_search_vector_store_request(
            vector_store_id="documents",
            query=query,
            vector_store_search_optional_params={},
            api_base=_BASE,
            litellm_logging_obj=_logging(),
            litellm_params={},
        )


def test_environment_credentials_and_reverse_proxy_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "environment-key")
    monkeypatch.setenv("QDRANT_API_BASE", "https://qdrant.example/prefix/")
    config: Final = QdrantVectorStoreConfig()
    assert config.get_complete_url(None, {}) == "https://qdrant.example/prefix"
    assert config.validate_environment({"X-Custom": "value"}, GenericLiteLLMParams()) == {
        "X-Custom": "value",
        "api-key": "environment-key",
        "Content-Type": "application/json",
    }
    assert config.validate_environment({}, GenericLiteLLMParams(api_key="explicit-key"))["api-key"] == "explicit-key"


def test_self_hosted_qdrant_without_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    config: Final = QdrantVectorStoreConfig()
    assert config.validate_environment({}, None) == {"Content-Type": "application/json"}
    assert config.get_complete_url("http://qdrant:6333/", {}) == "http://qdrant:6333"


@pytest.mark.parametrize("url", ["ftp://host", "https://user:password@host", "https://host?key=x", "https://host#x"])
def test_invalid_base_url_is_rejected(url: str) -> None:
    with pytest.raises(litellm.BadRequestError):
        QdrantVectorStoreConfig().get_complete_url(url, {})


def test_create_is_rejected_without_an_upstream_call() -> None:
    with pytest.raises(litellm.BadRequestError, match="search only"):
        litellm.vector_stores.create(custom_llm_provider="qdrant", name="documents")


@pytest.mark.asyncio
async def test_async_create_is_rejected_without_an_upstream_call() -> None:
    with pytest.raises(litellm.BadRequestError, match="search only"):
        await litellm.vector_stores.acreate(custom_llm_provider="qdrant", name="documents")
