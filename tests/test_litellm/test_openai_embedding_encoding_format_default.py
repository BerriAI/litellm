import json
from typing import Final

import httpx
import pytest
import respx

import litellm


def _mock_openai_embedding_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post("https://api.openai.com/v1/embeddings").mock(
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


def test_embedding_openai_omits_encoding_format_when_client_omits_it(respx_mock: respx.MockRouter) -> None:
    mock_route: Final = _mock_openai_embedding_route(respx_mock)

    response: Final = litellm.embedding(model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test")

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert "encoding_format" not in request_body
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
async def test_aembedding_openai_omits_encoding_format_when_client_omits_it(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    mock_route: Final = _mock_openai_embedding_route(respx_mock)

    response: Final = await litellm.aembedding(model="openai/text-embedding-3-small", input=["hello"], api_key="sk-test")

    request_body: Final = json.loads(mock_route.calls.last.request.read())
    assert "encoding_format" not in request_body
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
