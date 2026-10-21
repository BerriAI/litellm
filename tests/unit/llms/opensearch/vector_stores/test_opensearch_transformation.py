import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import MagicMock

import httpx
import openai
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.opensearch.vector_stores.transformation import OpenSearchVectorStoreConfig
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import VectorStoreSearchResponse

QUERY: Final = "How many days a year can I work abroad?"
QUERY_VECTOR: Final = [0.1, 0.2, 0.3]
OPENSEARCH: Final = "https://opensearch.internal:9200"
BASE_PARAMS: Final = {"api_base": OPENSEARCH, "litellm_embedding_model": "embedding-alias"}


class RecordingEmbeddingExecutor:
    def __init__(self) -> None:
        self.call: Final = MagicMock(return_value=EmbeddingResponse(data=[{"embedding": QUERY_VECTOR}]))

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)


def _sent(respond: MagicMock) -> tuple[httpx.Request, ...]:
    return tuple(call.args[0] for call in respond.call_args_list)


def _opensearch_hits(text_field: str, embedding_field: str) -> Mapping[str, object]:
    return {
        "took": 4,
        "timed_out": False,
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 0.581,
            "hits": [
                {
                    "_index": "litellm-qa-docs",
                    "_id": "1%3A0%3Aw6ftW6EB",
                    "_score": 0.581,
                    "_source": {
                        text_field: "Employees may work abroad for at most 37 calendar days per year.",
                        embedding_field: [0.5, 0.5, 0.5],
                        "filename": "hr-remote-work-policy.md",
                        "category": "hr",
                    },
                },
                {"_index": "litellm-qa-docs", "_id": "doc-2", "_score": 0.42, "_source": {"category": "facilities"}},
            ],
        },
    }


EXPECTED_RESULTS: Final = [
    {
        "score": 0.581,
        "content": [{"type": "text", "text": "Employees may work abroad for at most 37 calendar days per year."}],
        "file_id": "1%3A0%3Aw6ftW6EB",
        "filename": "hr-remote-work-policy.md",
        "attributes": {"filename": "hr-remote-work-policy.md", "category": "hr"},
    },
    {"score": 0.42, "content": [], "file_id": "doc-2", "filename": None, "attributes": {"category": "facilities"}},
]


async def _search(
    asynchronous: bool,
    respond: Callable[[httpx.Request], httpx.Response],
    executor: RecordingEmbeddingExecutor,
    kwargs: Mapping[str, object],
) -> VectorStoreSearchResponse:
    request_kwargs: Final = {
        **BASE_PARAMS,
        "vector_store_id": "litellm-qa-docs",
        "query": QUERY,
        "custom_llm_provider": "opensearch",
        "_direct_vector_store_embedding_executor": executor,
        **kwargs,
    }
    if asynchronous:
        return await litellm.vector_stores.asearch(
            **request_kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(respond))
        )
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        return litellm.vector_stores.search(**request_kwargs, client=HTTPHandler(client=transport))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "fields",
    [{}, {"opensearch_embedding_field": "passage_vector", "opensearch_text_field": "passage"}],
)
@pytest.mark.parametrize("api_base", [OPENSEARCH, "http://localhost:9200/", "https://search.docs.example/os/"])
@pytest.mark.asyncio
async def test_search_sends_an_unsigned_knn_query_and_returns_an_openai_results_page(
    asynchronous: bool, fields: Mapping[str, str], api_base: str
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    embedding_field: Final = fields.get("opensearch_embedding_field", "embedding")
    text_field: Final = fields.get("opensearch_text_field", "text")
    respond: Final = MagicMock(return_value=httpx.Response(200, json=_opensearch_hits(text_field, embedding_field)))

    response: Final = await _search(
        asynchronous,
        respond,
        executor,
        {**fields, "api_base": api_base, "max_num_results": 3},
    )

    assert response == {"object": "vector_store.search_results.page", "search_query": QUERY, "data": EXPECTED_RESULTS}
    sent: Final = _sent(respond)
    assert len(sent) == 1
    assert str(sent[0].url) == f"{api_base.rstrip('/')}/litellm-qa-docs/_search"
    assert json.loads(sent[0].content) == {
        "size": 3,
        "query": {"knn": {embedding_field: {"vector": QUERY_VECTOR, "k": 3}}},
        "_source": {"excludes": [embedding_field]},
    }
    assert sent[0].headers["content-type"] == "application/json"
    assert "authorization" not in sent[0].headers
    executor.call.assert_called_once_with("embedding-alias", QUERY, {})


@pytest.mark.asyncio
async def test_search_defaults_to_ten_results_like_openai() -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    response: Final = await _search(True, respond, RecordingEmbeddingExecutor(), {})

    assert response == {"object": "vector_store.search_results.page", "search_query": QUERY, "data": []}
    body: Final = json.loads(_sent(respond)[0].content)
    assert (body["size"], body["query"]["knn"]["embedding"]["k"]) == (10, 10)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_a_list_query_is_embedded_and_echoed_as_one_string(asynchronous: bool) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    response: Final = await _search(asynchronous, respond, executor, {"query": ["remote work", "abroad"]})

    assert response["search_query"] == "remote work abroad"
    executor.call.assert_called_once_with("embedding-alias", "remote work abroad", {})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [403, 404, 429, 500])
@pytest.mark.asyncio
async def test_opensearch_error_keeps_its_status(asynchronous: bool, status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"status": status, "error": {"reason": "upstream refused", "type": "x"}})

    with pytest.raises(openai.APIStatusError) as error:
        await _search(asynchronous, respond, RecordingEmbeddingExecutor(), {})

    assert error.value.status_code == status


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"filters": {"type": "eq", "key": "category", "value": "hr"}}, "filters"),
        ({"ranking_options": {"score_threshold": 0.5}}, "ranking_options"),
        ({"rewrite_query": True}, "rewrite_query"),
        ({"max_num_results": 0}, "max_num_results"),
        ({"max_num_results": 51}, "max_num_results"),
        ({"max_num_results": "ten"}, "Invalid vector store search parameters"),
        ({"litellm_embedding_model": None}, "litellm_embedding_model"),
        ({"opensearch_text_feild": "passage"}, "opensearch_text_feild"),
        ({"opensearch_text_field": 7}, "opensearch_text_field"),
        ({"extra_body": {"min_score": 0.5}}, "extra_body"),
        ({"api_base": None}, "api_base"),
        ({"api_base": "opensearch.internal:9200"}, "api_base"),
    ],
)
@pytest.mark.asyncio
async def test_unsupported_search_is_rejected_before_embedding_or_any_request(
    overrides: Mapping[str, object], message: str
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    with pytest.raises(litellm.BadRequestError, match=message) as error:
        await _search(True, respond, executor, overrides)

    assert error.value.status_code == 400
    respond.assert_not_called()
    executor.call.assert_not_called()


@pytest.mark.asyncio
async def test_no_op_values_of_unsupported_params_are_accepted() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_opensearch_hits("text", "embedding"))

    response: Final = await _search(
        True, respond, RecordingEmbeddingExecutor(), {"filters": {}, "ranking_options": {}, "rewrite_query": False}
    )

    assert response["data"] == EXPECTED_RESULTS


def test_create_is_rejected_because_the_store_is_search_only() -> None:
    with pytest.raises(litellm.BadRequestError, match="search-only"):
        litellm.vector_stores.create(custom_llm_provider="opensearch")


@pytest.mark.parametrize(
    "response,status",
    [
        (httpx.Response(403, json={"status": 403, "error": {"type": "Forbidden"}}), 403),
        (httpx.Response(200, json={"took": 1}), 502),
        (httpx.Response(200, text="<html>gateway</html>"), 502),
    ],
)
def test_response_without_search_hits_is_an_error_not_an_empty_page(response: httpx.Response, status: int) -> None:
    logging: Final = Logging(
        model="",
        messages=[],
        stream=False,
        call_type="vector_store_search",
        start_time=datetime(2026, 10, 20, tzinfo=timezone.utc),
        litellm_call_id="opensearch-test",
        function_id="opensearch-test",
    )

    with pytest.raises(BaseLLMException) as error:
        OpenSearchVectorStoreConfig().transform_search_vector_store_response(response, logging)

    assert error.value.status_code == status
