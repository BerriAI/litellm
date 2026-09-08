import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.mongodb.vector_stores.transformation import MongoDBVectorStoreConfig
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import VectorStoreSearchOptionalRequestParams

BASE_PARAMS: Final = {
    "api_base": "https://sidecar.example/prefix",
    "api_key": "test-sidecar-key",
    "litellm_embedding_model": "embedding-alias",
    "mongodb_database": "policies",
    "mongodb_collection": "documents",
}
RESULT: Final = {
    "object": "vector_store.search_results.page",
    "search_query": "travel policy",
    "data": [
        {"score": 0.9, "file_id": "123", "filename": "123", "content": [{"type": "text", "text": "Use code BLUE-42"}]}
    ],
}


class RecordingEmbeddingExecutor:
    def __init__(self) -> None:
        self.call: Final = MagicMock(return_value=EmbeddingResponse(data=[{"embedding": [0.1, 0.2, 0.3]}]))

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit,candidates", [(None, 100), (1, 100), (50, 500)])
@pytest.mark.asyncio
async def test_search_preserves_embedding_and_http_contract(
    asynchronous: bool, limit: int | None, candidates: int
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    config: Final = MongoDBVectorStoreConfig(executor)
    params: Final = {
        **BASE_PARAMS,
        "mongodb_text_field": "metadata.body",
        "mongodb_embedding_field": "stored_vector",
        "litellm_embedding_config": {"dimensions": 3},
        "timeout": 0.75,
    }
    kwargs: Final = {
        "vector_store_id": "exact index",
        "query": ["travel", "policy"],
        "vector_store_search_optional_params": {"max_num_results": limit},
        "api_base": BASE_PARAMS["api_base"],
        "litellm_logging_obj": MagicMock(),
        "litellm_params": params,
    }
    if asynchronous:
        url, body = await config.atransform_search_vector_store_request(**kwargs)
    else:
        url, body = config.transform_search_vector_store_request(**kwargs)
    assert url == "https://sidecar.example/prefix/v1/vector_stores/exact%20index/search"
    assert body == {
        "query": "travel policy",
        "query_vector": (0.1, 0.2, 0.3),
        "mongodb_database": "policies",
        "mongodb_collection": "documents",
        "mongodb_text_field": "metadata.body",
        "mongodb_embedding_field": "stored_vector",
        "mongodb_num_candidates": candidates,
        "max_num_results": limit or 10,
        "timeout_ms": 750,
    }
    executor.call.assert_called_once_with("embedding-alias", "travel policy", {"dimensions": 3})
    assert config.transform_search_vector_store_response(httpx.Response(200, json=RESULT), MagicMock()) == RESULT


@pytest.mark.parametrize(
    "query,overrides,options",
    [
        ("", {}, {}),
        ("  ", {}, {}),
        ("x" * 32_001, {}, {}),
        ("travel", {"litellm_embedding_model": None}, {}),
        ("travel", {"mongodb_database": None}, {}),
        ("travel", {"mongodb_collection": None}, {}),
        ("travel", {"mongodb_connection_string": "mongodb://obsolete-secret"}, {}),
        ("travel", {"mongodb_filter": {"private": True}}, {}),
        ("travel", {"mongodb_num_candidates": 9}, {}),
        ("travel", {"mongodb_num_candidates": 10_001}, {}),
        ("travel", {}, {"max_num_results": 0}),
        ("travel", {}, {"max_num_results": 51}),
        ("travel", {}, {"filters": {}}),
        ("travel", {}, {"ranking_options": {}}),
        ("travel", {}, {"rewrite_query": False}),
    ],
)
def test_invalid_search_is_rejected_before_embedding(
    query: str, overrides: Mapping[str, object], options: VectorStoreSearchOptionalRequestParams
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    config: Final = MongoDBVectorStoreConfig(executor)
    with pytest.raises(litellm.BadRequestError) as error:
        config.transform_search_vector_store_request(
            vector_store_id="policy_index",
            query=query,
            vector_store_search_optional_params=options,
            api_base=BASE_PARAMS["api_base"],
            litellm_logging_obj=MagicMock(),
            litellm_params={**BASE_PARAMS, **overrides},
        )
    assert "obsolete-secret" not in str(error.value)
    executor.call.assert_not_called()


@pytest.mark.parametrize(
    "status,body,error_type",
    [
        (400, {"error": {"message": "Index is not queryable"}}, litellm.BadRequestError),
        (401, {}, litellm.AuthenticationError),
        (408, {}, litellm.Timeout),
        (503, {}, litellm.ServiceUnavailableError),
        (200, {}, litellm.ServiceUnavailableError),
        (200, {**RESULT, "data": [{"score": "wrong"}]}, litellm.ServiceUnavailableError),
        (0, {}, litellm.Timeout),
        (-1, {}, litellm.BadRequestError),
        (200, RESULT, None),
    ],
)
def test_public_sdk_preserves_http_errors_response_and_timeout(
    status: int, body: Mapping[str, object], error_type: type[Exception] | None
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    if status == -1:
        with pytest.raises(litellm.BadRequestError, match="search-only"):
            litellm.vector_stores.create(custom_llm_provider="mongodb")
        executor.call.assert_not_called()
        return

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://sidecar.example/prefix/v1/vector_stores/policy_index/search"
        assert request.headers["authorization"] == "Bearer test-sidecar-key"
        assert request.extensions["timeout"]["read"] == 0.75
        payload: Final = json.loads(request.content)
        assert payload["timeout_ms"] == 750
        assert payload["query_vector"] == [0.1, 0.2, 0.3]
        if status == 0:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(status, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client: Final = HTTPHandler(client=transport)
        if error_type is not None:
            with pytest.raises(error_type):
                litellm.vector_stores.search(
                    vector_store_id="policy_index",
                    query="travel policy",
                    custom_llm_provider="mongodb",
                    _direct_vector_store_embedding_executor=executor,
                    client=client,
                    timeout=0.75,
                    **BASE_PARAMS,
                )
        else:
            result: Final = litellm.vector_stores.search(
                vector_store_id="policy_index",
                query="travel policy",
                custom_llm_provider="mongodb",
                _direct_vector_store_embedding_executor=executor,
                client=client,
                timeout=0.75,
                **BASE_PARAMS,
            )
            assert result == RESULT
    executor.call.assert_called_once_with("embedding-alias", "travel policy", {})
