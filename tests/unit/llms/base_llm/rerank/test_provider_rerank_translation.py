import json
from typing import Final, Mapping, cast

import httpx
import pytest
import respx
from pydantic import JsonValue
from respx import MockRouter
from typing_extensions import ReadOnly, TypedDict

import litellm


class _Kwargs(TypedDict, total=False):
    model: ReadOnly[str]
    api_key: ReadOnly[str]
    aws_access_key_id: ReadOnly[str]
    aws_secret_access_key: ReadOnly[str]
    aws_region_name: ReadOnly[str]


class _Case(TypedDict):
    id: ReadOnly[str]
    provider: ReadOnly[str]
    kwargs: ReadOnly[_Kwargs]
    url: ReadOnly[str]
    expected_cost_zero: ReadOnly[bool]
    billed_units: ReadOnly[Mapping[str, int]]
    response_id: ReadOnly[str | None]


_AWS: Final[Mapping[str, str]] = {
    "aws_access_key_id": "AKIAFAKE",
    "aws_secret_access_key": "fakesecret",
    "aws_region_name": "us-west-2",
}


def _bedrock_arn(model_id: str) -> str:
    return f"bedrock/arn:aws:bedrock:us-west-2::foundation-model/{model_id}"


_CASES: Final[tuple[_Case, ...]] = (
    {
        "id": "jina_reranker",
        "provider": "cohere",
        "kwargs": {"model": "jina_ai/jina-reranker-v2-base-multilingual", "api_key": "jina-offline"},
        "url": "https://api.jina.ai/v1/rerank",
        "expected_cost_zero": False,
        "billed_units": {"total_tokens": 4},
        "response_id": "rerank-offline",
    },
    {
        "id": "bedrock_amazon_rerank",
        "provider": "bedrock",
        "kwargs": cast(_Kwargs, {"model": _bedrock_arn("amazon.rerank-v1:0"), **dict(_AWS)}),
        "url": "https://bedrock-agent-runtime.us-west-2.amazonaws.com/rerank",
        "expected_cost_zero": False,
        "billed_units": {"search_units": 1},
        "response_id": "rerank-offline",
    },
    {
        "id": "bedrock_cohere_rerank",
        "provider": "bedrock",
        "kwargs": cast(_Kwargs, {"model": _bedrock_arn("cohere.rerank-v3-5:0"), **dict(_AWS)}),
        "url": "https://bedrock-agent-runtime.us-west-2.amazonaws.com/rerank",
        "expected_cost_zero": False,
        "billed_units": {"search_units": 1},
        "response_id": "rerank-offline",
    },
    {
        "id": "nvidia_nim_rerank",
        "provider": "nvidia_nim",
        "kwargs": {"model": "nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2", "api_key": "nvapi-offline"},
        "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        "expected_cost_zero": True,
        "billed_units": {"total_tokens": 4},
        "response_id": None,
    },
)


def _canned_response(case: _Case) -> httpx.Response:
    if case["provider"] == "cohere":
        return httpx.Response(
            200,
            json={
                "id": "rerank-offline",
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.4},
                ],
                "usage": {"total_tokens": 4},
            },
        )
    if case["provider"] == "bedrock":
        return httpx.Response(
            200,
            json={
                "id": "rerank-offline",
                "results": [
                    {"index": 0, "relevanceScore": 0.95},
                    {"index": 1, "relevanceScore": 0.4},
                ],
                "usage": {"search_units": 1},
            },
        )
    return httpx.Response(
        200,
        json={
            "rankings": [
                {"index": 0, "logit": 0.95},
                {"index": 1, "logit": 0.4},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )


@pytest.fixture(autouse=True)
def _httpx_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


def _request_body(route: respx.Route) -> Mapping[str, JsonValue]:
    return json.loads(route.calls.last.request.content)


def _assert_translated_request(case: _Case, body: Mapping[str, JsonValue]) -> None:
    if case["provider"] == "bedrock":
        queries: Final = body["queries"]
        assert queries == [{"textQuery": {"text": "hello"}, "type": "TEXT"}]
        config: Final = body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]
        assert config["modelArn"].endswith((".rerank-v1:0", ".rerank-v3-5:0"))
        sources: Final = body["sources"]
        assert len(sources) == 2
    elif case["provider"] == "nvidia_nim":
        assert body["model"] == "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        assert body["query"] == {"text": "hello"}
        assert body["passages"] == [{"text": "hello"}, {"text": "world"}]
        assert body["top_k"] == 2
    else:
        assert body["model"] == "jina-reranker-v2-base-multilingual"
        assert body["query"] == "hello"
        assert body["documents"] == ["hello", "world"]
        assert body["top_n"] == 2


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("sync_mode", (True, False))
@pytest.mark.asyncio
async def test_basic_rerank(case: _Case, sync_mode: bool, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(case["url"]).mock(return_value=_canned_response(case))
    if sync_mode:
        response: Final = litellm.rerank(
            **dict(case["kwargs"]),
            query="hello",
            documents=["hello", "world"],
            top_n=2,
        )
    else:
        response: Final = await litellm.arerank(
            **dict(case["kwargs"]),
            query="hello",
            documents=["hello", "world"],
            top_n=2,
        )
    body: Final = _request_body(route)
    _assert_translated_request(case, body)
    assert route.call_count == 1
    assert isinstance(response.id, str)
    if case["response_id"] is not None:
        assert response.id == case["response_id"]
    assert response.meta["billed_units"] == case["billed_units"]
    assert len(response.results) == 2
    assert response.results[0]["index"] == 0
    assert response.results[0]["relevance_score"] == 0.95
    assert response.results[1]["index"] == 1
    assert response.results[1]["relevance_score"] == 0.4
    if case["provider"] == "nvidia_nim":
        assert response.results[0]["document"] == {"text": "hello"}
        assert response.results[1]["document"] == {"text": "world"}
    cost: Final = response._hidden_params["response_cost"]
    if case["expected_cost_zero"]:
        assert cost == 0.0
    else:
        assert cost > 0
