import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.rerank import RerankBilledUnits, RerankResponse

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


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", (True, False), ids=("sync", "async"))
@pytest.mark.parametrize(
    ("billed_units", "reported_cost"),
    (
        pytest.param({"search_units": 2, "total_tokens": 2, "cost": 0.001}, 0.001, id="charged"),
        pytest.param({"search_units": 2, "total_tokens": 2, "cost": 0.0}, 0.0, id="free"),
        pytest.param({"search_units": 2, "total_tokens": 2}, None, id="absent-cost"),
        pytest.param({"search_units": 2, "total_tokens": 2, "cost": None}, None, id="null-cost"),
    ),
)
async def test_scaleway_rerank_preserves_openrouter_reported_cost(
    sync_mode: bool,
    billed_units: RerankBilledUnits,
    reported_cost: float | None,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setattr(litellm, "cache", None)
    price_per_token: Final = 0.002
    monkeypatch.setitem(
        litellm.model_cost,
        "scaleway/cohere/rerank-v3.5",
        {"input_cost_per_token": price_per_token, "litellm_provider": "scaleway", "mode": "rerank"},
    )
    route: Final = respx_mock.post("https://openrouter.ai/api/v1/rerank")
    route.return_value = httpx.Response(
        200,
        json={
            "id": "openrouter-rerank-cost-test",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": billed_units,
        },
    )

    response: Final = (
        litellm.rerank(
            model="scaleway/cohere/rerank-v3.5",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-openrouter-key",
            query="the customer disputed the deployment terms",
            documents=["the customer disputed the deployment terms"],
            top_n=1,
        )
        if sync_mode
        else await litellm.arerank(
            model="scaleway/cohere/rerank-v3.5",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-openrouter-key",
            query="the customer disputed the deployment terms",
            documents=["the customer disputed the deployment terms"],
            top_n=1,
        )
    )

    assert isinstance(response, RerankResponse)
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "model": "cohere/rerank-v3.5",
        "query": "the customer disputed the deployment terms",
        "documents": ["the customer disputed the deployment terms"],
        "top_n": 1,
    }
    assert response.meta == {"billed_units": billed_units, "tokens": {}}
    expected_cost: Final = 2 * price_per_token if reported_cost is None else reported_cost
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)
