import json
from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.together_ai.rerank.handler import TogetherAIRerank
from litellm.llms.together_ai.rerank.transformation import TogetherAIRerankConfig
from litellm.types.rerank import RerankBilledUnits

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


@pytest.mark.parametrize(
    ("usage_fields", "expected_billed_units"),
    (
        pytest.param({}, {}, id="missing-usage"),
        pytest.param({"usage": {}}, {}, id="empty-usage"),
        pytest.param({"usage": {"search_units": 1, "cost": None}}, {"search_units": 1, "cost": None}, id="null-cost"),
        pytest.param({"usage": {"search_units": 1, "cost": 0.0}}, {"search_units": 1, "cost": 0.0}, id="free"),
        pytest.param(
            {
                "usage": {
                    "search_units": 2,
                    "total_tokens": 7,
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cost": 0.001,
                    "extra": 9,
                }
            },
            {"search_units": 2, "total_tokens": 7, "cost": 0.001},
            id="charged-with-token-counts",
        ),
    ),
)
def test_together_ai_rerank_preserves_billed_units(
    usage_fields: Mapping[str, object], expected_billed_units: RerankBilledUnits
) -> None:
    response: Final = TogetherAIRerankConfig().transform_response(
        {"results": [{"index": 0, "relevance_score": 0.9}], **usage_fields}
    )

    assert response.meta is not None
    assert response.meta["billed_units"] == expected_billed_units
