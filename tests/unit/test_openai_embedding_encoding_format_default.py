import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from openai.types import CreateEmbeddingResponse

_SDK_EMBEDDING_MODELS: Final = (
    "openai/sdk-compat",
    "mistral/sdk-compat",
    "fireworks_ai/accounts/fireworks/models/sdk-compat",
    "together_ai/sdk-compat",
    "nvidia_nim/sdk-compat",
)


def _mock_openai_embedding_route(
    respx_mock: respx.MockRouter, api_base: str = "https://api.openai.com/v1"
) -> respx.Route:
    return respx_mock.post(f"{api_base}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )


@pytest.fixture(autouse=True)
def clear_default_encoding_format_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", raising=False)


@pytest.mark.parametrize("model", _SDK_EMBEDDING_MODELS)
def test_embedding_sdk_providers_omit_encoding_format_when_client_omits_it(
    respx_mock: respx.MockRouter, model: str
) -> None:
    provider, upstream_model = model.split("/", 1)
    api_base: Final = f"https://{provider.replace('_', '-')}.example/v1"
    mock_route: Final = _mock_openai_embedding_route(respx_mock, api_base)

    response: Final = litellm.embedding(model=model, input=["hello"], api_key="sk-test", api_base=api_base)

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert "encoding_format" not in request_body
    assert request_body["model"] == upstream_model
    assert request_body["input"] == ["hello"]
    assert mock_route.calls.last.request.headers["authorization"] == "Bearer sk-test"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_embedding_openai_forwards_explicit_encoding_format(respx_mock: respx.MockRouter) -> None:
    mock_route: Final = _mock_openai_embedding_route(respx_mock)

    litellm.embedding(
        model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test", encoding_format="base64"
    )

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert request_body["encoding_format"] == "base64"


def test_embedding_openai_explicit_encoding_format_wins_over_env_var(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", "float")
    mock_route: Final = _mock_openai_embedding_route(respx_mock)

    litellm.embedding(
        model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test", encoding_format="base64"
    )

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert request_body["encoding_format"] == "base64"


@pytest.mark.parametrize("env_value", ["float", "base64"])
def test_embedding_openai_env_var_sets_default_encoding_format(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, env_value: str
) -> None:
    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", env_value)
    mock_route: Final = _mock_openai_embedding_route(respx_mock)

    litellm.embedding(model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test")

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert request_body["encoding_format"] == env_value


@pytest.mark.parametrize("env_none", ["none", "NONE", " none "])
def test_embedding_openai_env_none_omits_encoding_format(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, env_none: str
) -> None:
    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", env_none)
    mock_route: Final = _mock_openai_embedding_route(respx_mock)

    litellm.embedding(model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test")

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert "encoding_format" not in request_body


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _SDK_EMBEDDING_MODELS)
async def test_aembedding_sdk_providers_omit_encoding_format_when_client_omits_it(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    provider, upstream_model = model.split("/", 1)
    api_base: Final = f"https://{provider.replace('_', '-')}.example/v1"
    mock_route: Final = _mock_openai_embedding_route(respx_mock, api_base)

    response: Final = await litellm.aembedding(model=model, input=["hello"], api_key="sk-test", api_base=api_base)

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert "encoding_format" not in request_body
    assert request_body["model"] == upstream_model
    assert request_body["input"] == ["hello"]
    assert mock_route.calls.last.request.headers["authorization"] == "Bearer sk-test"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_embedding_openai_omitted_encoding_format_maps_provider_errors(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx_mock.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "42", "x-should-retry": "false"},
            json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
        )
    )

    with pytest.raises(litellm.RateLimitError) as exc_info:
        litellm.embedding(
            model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test", max_retries=0
        )

    assert int(exc_info.value.litellm_response_headers["retry-after"]) == 42


@pytest.mark.parametrize(
    ("model", "url", "extra_params"),
    (
        ("openai/text-embedding-3-small", "https://api.openai.com/v1/embeddings", {}),
        (
            "azure/migration-embedding-deployment",
            "https://migration.openai.azure.com/openai/deployments/migration-embedding-deployment/embeddings",
            {"api_base": "https://migration.openai.azure.com", "api_version": "2024-10-21"},
        ),
    ),
)
def test_embedding_response_preserves_rate_limit_headers(
    respx_mock: respx.MockRouter, model: str, url: str, extra_params: dict[str, str]
) -> None:
    respx_mock.post(url__startswith=url).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "9",
                "x-ratelimit-remaining-tokens": "99",
            },
        )
    )

    response: Final = litellm.embedding(model=model, input=["Hello world"], api_key="sk-test", **extra_params)

    additional_headers: Final = response._hidden_params["additional_headers"]
    assert {
        name: additional_headers[name]
        for name in (
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "llm_provider-x-ratelimit-remaining-requests",
            "llm_provider-x-ratelimit-remaining-tokens",
        )
    } == {
        "x-ratelimit-remaining-requests": "9",
        "x-ratelimit-remaining-tokens": "99",
        "llm_provider-x-ratelimit-remaining-requests": "9",
        "llm_provider-x-ratelimit-remaining-tokens": "99",
    }


def _openai_sdk_response_keys(route: respx.Route) -> frozenset[str]:
    return frozenset(dict(CreateEmbeddingResponse.model_validate(route.return_value.json())))


def test_openai_embedding_returns_sdk_shaped_vectors(respx_mock: respx.MockRouter) -> None:
    route: Final = _mock_openai_embedding_route(respx_mock)

    response: Final = litellm.embedding(
        model="openai/text-embedding-3-small",
        input=["first", "second"],
        api_key="sk-test",
    )

    assert frozenset(dict(response)) - {"_response_ms"} == _openai_sdk_response_keys(route)
    assert [row["embedding"] for row in response.data] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert len(route.calls) == 1


def test_openai_embedding_forwards_dimensions(respx_mock: respx.MockRouter) -> None:
    route: Final = _mock_openai_embedding_route(respx_mock)

    response: Final = litellm.embedding(
        model="openai/text-embedding-3-small",
        input=["first", "second"],
        api_key="sk-test",
        dimensions=5,
    )

    body: Final = json.loads(route.calls.last.request.read())
    assert body["dimensions"] == 5
    assert body["input"] == ["first", "second"]
    assert frozenset(dict(response)) - {"_response_ms"} == _openai_sdk_response_keys(route)
    assert len(response.data) == 2


@pytest.mark.asyncio
async def test_aembedding_openai_returns_the_mocked_vector(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _mock_openai_embedding_route(respx_mock)

    response: Final = await litellm.aembedding(
        model="openai/text-embedding-3-small",
        input=["good morning from litellm", "this is another item"],
        api_key="sk-test",
    )
    cost: Final = litellm.completion_cost(
        completion_response=response,
        custom_cost_per_token={"input_cost_per_token": 0.001, "output_cost_per_token": 0.0},
    )

    assert [row["embedding"] for row in response.data] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert cost == pytest.approx(0.002)
