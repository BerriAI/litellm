import json
from typing import Final

import httpx
import pytest
import respx

import litellm

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
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
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
