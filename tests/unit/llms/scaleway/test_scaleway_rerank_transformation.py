import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.caching.caching import Cache
from litellm.litellm_core_utils.model_cost_table import publish
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.rerank import RerankResponse

SCALEWAY_RERANK_BODY = {
    "id": "rerank-a89e6d7b8b97492ea81569c65fbfff49",
    "model": "qwen3-embedding-8b",
    "usage": {"total_tokens": 99},
    "results": [
        {
            "index": 1,
            "document": {"text": "Oceans can be sorted by size: Pacific, Atlantic, Indian", "multi_modal": None},
            "relevance_score": 0.6456239223480225,
        },
        {
            "index": 0,
            "document": {"text": "The Pacific is approximately 165 million km²", "multi_modal": None},
            "relevance_score": 0.6059925556182861,
        },
    ],
}

DOCUMENTS = ["The Pacific is approximately 165 million km²", "Oceans can be sorted by size: Pacific, Atlantic, Indian"]


def test_scaleway_rerank_posts_to_the_documented_endpoint(respx_mock: respx.MockRouter, monkeypatch):
    monkeypatch.delenv("SCALEWAY_API_BASE", raising=False)
    route = respx_mock.post("https://api.scaleway.ai/v1/rerank")
    route.return_value = httpx.Response(200, json=SCALEWAY_RERANK_BODY)

    response = litellm.rerank(
        model="scaleway/qwen3-embedding-8b",
        query="What is the biggest area of water on earth ?",
        documents=DOCUMENTS,
        top_n=2,
        api_key="scw-key",
    )

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer scw-key"
    assert json.loads(request.content) == {
        "model": "qwen3-embedding-8b",
        "query": "What is the biggest area of water on earth ?",
        "documents": DOCUMENTS,
        "top_n": 2,
    }
    assert [r["index"] for r in response.results] == [1, 0]
    assert response.results[0]["relevance_score"] == pytest.approx(0.6456239223480225)
    assert response.results[0]["document"]["text"].startswith("Oceans")
    assert response.id == SCALEWAY_RERANK_BODY["id"]
    assert response.meta["billed_units"]["total_tokens"] == 99


def test_scaleway_rerank_reads_the_key_from_scw_secret_key(respx_mock: respx.MockRouter, monkeypatch):
    monkeypatch.setenv("SCW_SECRET_KEY", "env-scw-key")
    route = respx_mock.post("https://api.scaleway.ai/v1/rerank")
    route.return_value = httpx.Response(200, json=SCALEWAY_RERANK_BODY)

    litellm.rerank(model="scaleway/qwen3-embedding-8b", query="q", documents=DOCUMENTS)

    assert route.calls[0].request.headers["authorization"] == "Bearer env-scw-key"


def test_scaleway_rerank_honors_api_base(respx_mock: respx.MockRouter):
    route = respx_mock.post("https://scw.example/v1/rerank")
    route.return_value = httpx.Response(200, json=SCALEWAY_RERANK_BODY)

    litellm.rerank(
        model="scaleway/qwen3-embedding-8b",
        query="q",
        documents=DOCUMENTS,
        api_key="scw-key",
        api_base="https://scw.example/v1/",
    )

    assert route.called


def test_scaleway_rerank_does_not_send_return_documents(respx_mock: respx.MockRouter):
    """The Scaleway API has no such field, so it must not reach the request body."""
    route = respx_mock.post("https://api.scaleway.ai/v1/rerank")
    route.return_value = httpx.Response(200, json=SCALEWAY_RERANK_BODY)

    litellm.rerank(
        model="scaleway/qwen3-embedding-8b",
        query="q",
        documents=DOCUMENTS,
        return_documents=True,
        api_key="scw-key",
    )

    assert "return_documents" not in json.loads(route.calls[0].request.content)


def test_scaleway_rerank_without_a_key_names_the_env_var(monkeypatch):
    monkeypatch.delenv("SCW_SECRET_KEY", raising=False)

    with pytest.raises(litellm.APIConnectionError, match="SCW_SECRET_KEY"):
        litellm.rerank(model="scaleway/qwen3-embedding-8b", query="q", documents=DOCUMENTS)


def test_scaleway_rerank_caller_headers_cannot_replace_the_provider_key(respx_mock: respx.MockRouter):
    route = respx_mock.post("https://api.scaleway.ai/v1/rerank")
    route.return_value = httpx.Response(200, json=SCALEWAY_RERANK_BODY)

    litellm.rerank(
        model="scaleway/qwen3-embedding-8b",
        query="q",
        documents=DOCUMENTS,
        api_key="scw-key",
        headers={"Authorization": "Bearer caller-key", "x-trace": "abc"},
    )

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer scw-key"
    assert request.headers["x-trace"] == "abc"


@pytest.mark.asyncio
async def test_scaleway_arerank_posts_to_the_documented_endpoint():
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=httpx.Response(200, json=SCALEWAY_RERANK_BODY))

    response = await litellm.arerank(
        model="scaleway/qwen3-embedding-8b", query="q", documents=DOCUMENTS, api_key="scw-key", client=client
    )

    assert client.post.await_args.kwargs["url"] == "https://api.scaleway.ai/v1/rerank"
    assert client.post.await_args.kwargs["headers"]["authorization"] == "Bearer scw-key"
    assert [r["index"] for r in response.results] == [1, 0]


UNMAPPED_RERANK_MODEL: Final = "scaleway/test-unmapped-reranker"
REPORTED_USAGE: Final = {"search_units": 1, "cost": 0.002, "is_byok": False}
OPENROUTER_RERANK_BODY: Final = {
    "id": "gen-rerank-1",
    "model": "cohere/rerank-v3.5",
    "results": [
        {"index": 1, "relevance_score": 0.9, "document": {"text": DOCUMENTS[1]}},
        {"index": 0, "relevance_score": 0.1, "document": {"text": DOCUMENTS[0]}},
    ],
    "usage": REPORTED_USAGE,
}
UNREPORTED_COST_BODY: Final = {**OPENROUTER_RERANK_BODY, "usage": {"search_units": 1}}


async def _rerank_via_router(
    sync_mode: bool,
    body: Mapping[str, object],
    respx_mock: respx.MockRouter,
    model: str = UNMAPPED_RERANK_MODEL,
    pricing: Mapping[str, float] = MappingProxyType({}),
) -> RerankResponse:
    rerank_args: Final = {
        "model": model,
        "query": "q",
        "documents": DOCUMENTS,
        "top_n": 2,
        "api_key": "router-key",
        "api_base": "https://router.example/api/v1",
        **pricing,
    }
    if sync_mode:
        respx_mock.post("https://router.example/api/v1/rerank").return_value = httpx.Response(200, json=body)
        synced: Final = litellm.rerank(**rerank_args)
        assert isinstance(synced, RerankResponse)
        return synced
    client: Final = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=httpx.Response(200, json=body))
    awaited: Final = await litellm.arerank(**rerank_args, client=client)
    assert isinstance(awaited, RerankResponse)
    return awaited


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_scaleway_rerank_bills_and_returns_the_provider_reported_usage(
    sync_mode: bool, respx_mock: respx.MockRouter
) -> None:
    assert UNMAPPED_RERANK_MODEL not in litellm.model_cost
    response: Final = await _rerank_via_router(sync_mode, OPENROUTER_RERANK_BODY, respx_mock)

    assert response._hidden_params["response_cost"] == pytest.approx(REPORTED_USAGE["cost"])
    assert response.model_dump(by_alias=True)["usage"] == REPORTED_USAGE
    assert response.meta["billed_units"] == {"search_units": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_scaleway_rerank_without_a_reported_or_mapped_cost_reports_no_cost(
    sync_mode: bool, respx_mock: respx.MockRouter
) -> None:
    assert UNMAPPED_RERANK_MODEL not in litellm.model_cost
    response: Final = await _rerank_via_router(sync_mode, UNREPORTED_COST_BODY, respx_mock)

    assert response._hidden_params["response_cost"] is None
    assert response.model_dump(by_alias=True)["usage"] == {"search_units": 1}


def test_scaleway_rerank_cache_hit_costs_nothing_and_keeps_the_provider_usage(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    route: Final = respx_mock.post("https://router.example/api/v1/rerank")
    route.return_value = httpx.Response(200, json=OPENROUTER_RERANK_BODY)
    rerank_args: Final = {
        "model": UNMAPPED_RERANK_MODEL,
        "query": "q",
        "documents": DOCUMENTS,
        "api_key": "router-key",
        "api_base": "https://router.example/api/v1",
    }

    first: Final = litellm.rerank(**rerank_args)
    second: Final = litellm.rerank(**rerank_args)

    assert isinstance(first, RerankResponse)
    assert isinstance(second, RerankResponse)
    assert route.call_count == 1
    assert first._hidden_params["response_cost"] == pytest.approx(REPORTED_USAGE["cost"])
    assert second._hidden_params["response_cost"] == 0.0
    assert second.model_dump(by_alias=True)["usage"] == REPORTED_USAGE


REGISTERED_RERANK_MODEL: Final = "scaleway/test-registered-reranker"
TOKEN_REPORTED_BODY: Final = {**OPENROUTER_RERANK_BODY, "usage": {"total_tokens": 24, "cost": 4.8e-07}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_cost_per_token", "adjustment", "expected"),
    [
        (1e-06, {}, 24 * 1e-06),
        (0.0, {}, 0.0),
        (1e-06, {"cost_discount_config": {"scaleway": 0.5}}, 24 * 1e-06 * 0.5),
        (1e-06, {"cost_margin_config": {"scaleway": 0.1}}, 24 * 1e-06 * 1.1),
    ],
)
@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize("prefixed", [True, False])
async def test_scaleway_rerank_bills_a_registered_price_over_the_reported_cost(
    persist: bool,
    prefixed: bool,
    input_cost_per_token: float,
    adjustment: Mapping[str, object],
    expected: float,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for setting, value in adjustment.items():
        monkeypatch.setattr(litellm, setting, value)
    registered_name: Final = REGISTERED_RERANK_MODEL if prefixed else REGISTERED_RERANK_MODEL.removeprefix("scaleway/")
    litellm.register_model(
        {
            registered_name: {
                "litellm_provider": "scaleway",
                "mode": "rerank",
                "input_cost_per_token": input_cost_per_token,
            }
        },
        persist_across_reloads=persist,
    )

    response: Final = await _rerank_via_router(True, TOKEN_REPORTED_BODY, respx_mock, model=REGISTERED_RERANK_MODEL)

    assert response._hidden_params["response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_scaleway_arerank_bills_a_registered_price_over_the_reported_cost(respx_mock: respx.MockRouter) -> None:
    litellm.register_model(
        {REGISTERED_RERANK_MODEL: {"litellm_provider": "scaleway", "mode": "rerank", "input_cost_per_token": 1e-06}}
    )

    response: Final = await _rerank_via_router(False, TOKEN_REPORTED_BODY, respx_mock, model=REGISTERED_RERANK_MODEL)

    assert response._hidden_params["response_cost"] == pytest.approx(24 * 1e-06)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "pricing",
    [{"input_cost_per_token": 1e-06}, {"input_cost_per_token": 0.0}, {"input_cost_per_query": 0.005}],
)
async def test_scaleway_rerank_with_per_call_prices_keeps_the_zero_cost(
    sync_mode: bool, pricing: Mapping[str, float], respx_mock: respx.MockRouter
) -> None:
    assert UNMAPPED_RERANK_MODEL not in litellm.model_cost
    response: Final = await _rerank_via_router(sync_mode, OPENROUTER_RERANK_BODY, respx_mock, pricing=pricing)

    assert response._hidden_params["response_cost"] == 0.0


SHARED_RERANK_NAME: Final = "test-shared-reranker"


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize("provider_from_catalog", [True, False])
@pytest.mark.parametrize(
    ("registered_provider", "expected"),
    [("jina_ai", 4.8e-07), ("scaleway", 24 * 1e-06)],
)
async def test_scaleway_rerank_ignores_a_bare_name_registration_for_another_provider(
    persist: bool,
    provider_from_catalog: bool,
    registered_provider: str,
    expected: float,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if provider_from_catalog:
        monkeypatch.setitem(
            litellm.model_cost,
            SHARED_RERANK_NAME,
            {"litellm_provider": registered_provider, "mode": "rerank", "input_cost_per_token": 2e-06},
        )
    provider_metadata: Final = {} if provider_from_catalog else {"litellm_provider": registered_provider}
    litellm.register_model(
        {
            SHARED_RERANK_NAME: {
                **provider_metadata,
                "mode": "rerank",
                "input_cost_per_token": 1e-06,
            }
        },
        persist_across_reloads=persist,
    )

    response: Final = await _rerank_via_router(
        True, TOKEN_REPORTED_BODY, respx_mock, model=f"scaleway/{SHARED_RERANK_NAME}"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(expected)


CATALOG_RERANK_ROW: Final = "test-catalog-reranker"
CATALOG_RERANK_ALIAS: Final = f"cohere/{CATALOG_RERANK_ROW}"
TOKEN_USAGE_WITH_COST: Final = {"total_tokens": 24, "cost": 0.002}
SEARCH_USAGE_WITH_COST: Final = {"search_units": 1, "cost": 0.002}


@pytest.mark.asyncio
@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize(
    ("registration", "usage", "expected"),
    [
        (None, TOKEN_USAGE_WITH_COST, 0.002),
        (None, SEARCH_USAGE_WITH_COST, 0.002),
        ({"litellm_provider": "scaleway", "input_cost_per_token": 1e-06}, TOKEN_USAGE_WITH_COST, 0.0),
        ({"litellm_provider": "scaleway", "input_cost_per_token": 0.0}, SEARCH_USAGE_WITH_COST, 0.0),
        ({"litellm_provider": "jina_ai", "input_cost_per_token": 1e-06}, TOKEN_USAGE_WITH_COST, 0.002),
        ({"litellm_provider": None, "input_cost_per_token": 1e-06}, TOKEN_USAGE_WITH_COST, 0.002),
        ({"input_cost_per_token": 1e-06}, TOKEN_USAGE_WITH_COST, 0.002),
    ],
)
async def test_scaleway_rerank_honors_a_registration_merged_onto_a_published_catalog_row(
    persist: bool,
    registration: Mapping[str, object] | None,
    usage: Mapping[str, object],
    expected: float,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        CATALOG_RERANK_ROW,
        {"litellm_provider": "cohere", "mode": "rerank", "input_cost_per_query": 0.002, "input_cost_per_token": 0.0},
    )
    publish(litellm.model_cost)
    if registration is not None:
        litellm.register_model(
            {CATALOG_RERANK_ALIAS: {"mode": "rerank", **registration}}, persist_across_reloads=persist
        )
    assert CATALOG_RERANK_ALIAS not in litellm.model_cost

    response: Final = await _rerank_via_router(
        True, {**OPENROUTER_RERANK_BODY, "usage": usage}, respx_mock, model=f"scaleway/{CATALOG_RERANK_ALIAS}"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(expected)
