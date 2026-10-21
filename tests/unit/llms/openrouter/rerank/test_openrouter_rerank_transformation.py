"""
Unit tests for the OpenRouter rerank provider.

OpenRouter serves rerank models (e.g. `cohere/rerank-v3.5`) via
POST https://openrouter.ai/api/v1/rerank - all tests here are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.rerank.transformation import OpenRouterRerankConfig
from litellm.types.rerank import RerankResponse
from litellm.utils import ProviderConfigManager

OPENROUTER_RERANK_RESPONSE = {
    "id": "gen-rerank-1234567890-abc",
    "model": "cohere/rerank-v3.5",
    "provider": "Cohere",
    "results": [
        {"document": {"text": "Paris is the capital of France."}, "index": 0, "relevance_score": 0.98},
        {"document": {"text": "Berlin is the capital of Germany."}, "index": 1, "relevance_score": 0.12},
    ],
    "usage": {"search_units": 1, "total_tokens": 150},
}


class _RespxAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, router: respx.MockRouter) -> None:
        self.router = router

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.router.async_handler(request)


def _config() -> OpenRouterRerankConfig:
    return OpenRouterRerankConfig()


def _transform(payload: object, status_code: int = 200) -> RerankResponse:
    return _config().transform_rerank_response(
        model="cohere/rerank-v3.5",
        raw_response=httpx.Response(status_code, json=payload),
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
    )


def test_get_provider_rerank_config_returns_openrouter_config():
    config = ProviderConfigManager.get_provider_rerank_config(
        model="cohere/rerank-v3.5",
        provider=litellm.LlmProviders.OPENROUTER,
        api_base=None,
        present_version_params=[],
    )
    assert isinstance(config, OpenRouterRerankConfig)


def test_get_complete_url_defaults_to_openrouter_rerank_endpoint():
    assert _config().get_complete_url(None, "cohere/rerank-v3.5") == "https://openrouter.ai/api/v1/rerank"


def test_get_complete_url_respects_custom_api_base():
    assert (
        _config().get_complete_url("https://proxy.example.com/v1/", "cohere/rerank-v3.5")
        == "https://proxy.example.com/v1/rerank"
    )


def test_validate_environment_builds_auth_headers():
    headers = _config().validate_environment(headers={}, model="cohere/rerank-v3.5", api_key="sk-or-test")

    assert headers["Authorization"] == "Bearer sk-or-test"
    assert headers["content-type"] == "application/json"
    assert headers["HTTP-Referer"] == "https://litellm.ai"
    assert headers["X-Title"] == "liteLLM"


def test_validate_environment_falls_back_to_env_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")

    headers = _config().validate_environment(headers={}, model="cohere/rerank-v3.5")

    assert headers["Authorization"] == "Bearer sk-or-env"


def test_validate_environment_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _config().validate_environment(headers={}, model="cohere/rerank-v3.5")


def test_map_cohere_rerank_params_keeps_only_supported_params():
    mapped = _config().map_cohere_rerank_params(
        non_default_params={
            "query": "q",
            "documents": ["a", "b"],
            "top_n": 1,
            "return_documents": True,
            "max_chunks_per_doc": 3,
            "some_unknown_param": 1,
        },
        model="cohere/rerank-v3.5",
        drop_params=False,
        query="q",
        documents=["a", "b"],
        top_n=1,
    )

    assert mapped == {"query": "q", "documents": ["a", "b"], "top_n": 1}


def test_transform_rerank_request_strips_litellm_routing_prefix():
    request_data = _config().transform_rerank_request(
        model="openrouter/cohere/rerank-v3.5",
        optional_rerank_params={"query": "q", "documents": ["a", "b"], "top_n": 1},
        headers={},
    )

    assert request_data == {
        "model": "cohere/rerank-v3.5",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 1,
    }


@pytest.mark.parametrize("missing", ["query", "documents"])
def test_transform_rerank_request_requires_query_and_documents(missing: str):
    params = {"query": "q", "documents": ["a"]}
    del params[missing]

    with pytest.raises(ValueError, match=missing):
        _config().transform_rerank_request(model="cohere/rerank-v3.5", optional_rerank_params=params, headers={})


def test_transform_rerank_response_maps_openrouter_payload():
    response = _transform(OPENROUTER_RERANK_RESPONSE)

    assert response.id == "gen-rerank-1234567890-abc"
    assert response.results == [
        {"index": 0, "relevance_score": 0.98, "document": {"text": "Paris is the capital of France."}},
        {"index": 1, "relevance_score": 0.12, "document": {"text": "Berlin is the capital of Germany."}},
    ]
    assert response.meta == {"billed_units": {"search_units": 1, "total_tokens": 150}}
    assert response.hidden_params["provider"] == "Cohere"
    assert response.hidden_params["model"] == "cohere/rerank-v3.5"


def test_transform_rerank_response_wraps_string_documents():
    response = _transform(
        {
            "id": "gen-rerank-1",
            "results": [{"index": 1, "relevance_score": 0.5, "document": "plain string doc"}],
        }
    )

    assert response.results == [{"index": 1, "relevance_score": 0.5, "document": {"text": "plain string doc"}}]
    assert response.meta == {"billed_units": {}}


def test_transform_rerank_response_generates_an_id_when_the_provider_sends_none():
    response = _transform({"results": [{"index": 0, "relevance_score": 0.5}]})

    assert isinstance(response.id, str)
    assert response.id != ""


def test_transform_rerank_response_without_results_raises_value_error():
    with pytest.raises(ValueError, match="No results found"):
        _transform({"id": "gen-rerank-1"})


def test_transform_rerank_response_rejects_malformed_payloads():
    with pytest.raises(ValidationError):
        _transform({"results": [["leaked document text"]]})


def test_transform_rerank_response_non_200_raises_openrouter_exception():
    with pytest.raises(OpenRouterException, match="quota exceeded"):
        _transform({"error": {"message": "quota exceeded"}}, status_code=429)


def test_get_error_class_extracts_openrouter_error_message():
    error = _config().get_error_class(
        error_message=json.dumps({"error": {"message": "Invalid API key", "code": 401}}),
        status_code=401,
        headers={},
    )

    assert isinstance(error, OpenRouterException)
    assert error.message == "Invalid API key"
    assert error.status_code == 401


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openrouter_rerank_end_to_end(sync_mode: bool, respx_mock: respx.MockRouter, monkeypatch):
    """litellm.rerank(model="openrouter/...") routes to OpenRouter's /rerank endpoint."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    route = respx_mock.post("https://openrouter.ai/api/v1/rerank")
    route.return_value = httpx.Response(200, json=OPENROUTER_RERANK_RESPONSE)

    kwargs = {
        "model": "openrouter/cohere/rerank-v3.5",
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
        "top_n": 2,
    }
    if sync_mode:
        response = litellm.rerank(**kwargs)
    else:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        client = AsyncHTTPHandler(transport=_RespxAsyncTransport(respx_mock))
        response = await litellm.arerank(**kwargs, client=client)
        await client.client.aclose()

    assert route.called
    request_body = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "cohere/rerank-v3.5",
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
        "top_n": 2,
    }
    assert route.calls[0].request.headers["Authorization"] == "Bearer sk-or-test"

    assert response.results[0]["index"] == 0
    assert response.results[0]["relevance_score"] == 0.98
    assert response.meta["billed_units"] == {"search_units": 1, "total_tokens": 150}
