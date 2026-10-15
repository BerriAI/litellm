import json
from collections.abc import Iterator

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.together_ai.rerank.handler import TogetherAIRerank

API_BASE = "https://api.together.example/v1"
RERANKED = {
    "id": "rr-1",
    "results": [
        {"index": 1, "relevance_score": 0.9, "document": {"text": "Paris"}},
        {"index": 0, "relevance_score": 0.1},
    ],
    "usage": {"total_tokens": 7},
}
NON_OBJECT_BODIES = [7, "sensitive-document", [{"id": "sensitive-document"}]]


@pytest.fixture
def rerank_route(monkeypatch: pytest.MonkeyPatch) -> Iterator[respx.Route]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock as router:
        yield router.post(f"{API_BASE}/rerank")


def _rerank(*, is_async: bool):
    return TogetherAIRerank().rerank(
        model="Salesforce/Llama-Rank-V1",
        api_key="sk-test",
        api_base=API_BASE,
        query="capital of france",
        documents=["Berlin", {"text": "Paris"}],
        top_n=1,
        _is_async=is_async,
    )


def _assert_is_the_reranking(response: litellm.RerankResponse, route: respx.Route) -> None:
    assert response.id == "rr-1"
    assert response.results == [
        {"index": 1, "relevance_score": 0.9, "document": {"text": "Paris"}},
        {"index": 0, "relevance_score": 0.1},
    ]
    assert response.meta == {"billed_units": {"total_tokens": 7}, "tokens": {}}
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(route.calls.last.request.content) == {
        "model": "Salesforce/Llama-Rank-V1",
        "query": "capital of france",
        "top_n": 1,
        "documents": ["Berlin", {"text": "Paris"}],
        "return_documents": True,
    }


def test_rerank_returns_the_upstream_ranking(rerank_route: respx.Route):
    rerank_route.mock(return_value=httpx.Response(200, json=RERANKED))

    _assert_is_the_reranking(_rerank(is_async=False), rerank_route)


async def test_async_rerank_returns_the_upstream_ranking(rerank_route: respx.Route):
    rerank_route.mock(return_value=httpx.Response(200, json=RERANKED))

    _assert_is_the_reranking(await _rerank(is_async=True), rerank_route)


@pytest.mark.parametrize("payload", NON_OBJECT_BODIES)
def test_rerank_rejects_non_object_bodies(rerank_route: respx.Route, payload: object):
    rerank_route.mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(ValidationError) as exc_info:
        _rerank(is_async=False)

    assert "sensitive-document" not in str(exc_info.value)


@pytest.mark.parametrize("payload", NON_OBJECT_BODIES)
async def test_async_rerank_rejects_non_object_bodies(rerank_route: respx.Route, payload: object):
    rerank_route.mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(ValidationError) as exc_info:
        await _rerank(is_async=True)

    assert "sensitive-document" not in str(exc_info.value)


def test_rerank_without_results_is_a_value_error(rerank_route: respx.Route):
    rerank_route.mock(return_value=httpx.Response(200, json={"id": "rr-1"}))

    with pytest.raises(ValueError, match="No results found"):
        _rerank(is_async=False)
