import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.ollama.completion.handler import ollama_aembeddings, ollama_embeddings
from litellm.types.utils import EmbeddingResponse


@pytest.fixture
def mock_response_data():
    return {
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "prompt_eval_count": 5,
    }


@pytest.fixture
def mock_embedding_response():
    return EmbeddingResponse(object="", data=[], model="", usage=None)


@pytest.fixture
def mock_encoding():
    mock = MagicMock()
    mock.encode.return_value = [0] * 5
    return mock


def test_ollama_embeddings(mock_response_data, mock_embedding_response, mock_encoding):
    with (
        patch("litellm.module_level_client.post") as mock_post,
        patch("litellm.OllamaConfig.get_config", return_value={"truncate": 512}),
    ):

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response

        response = ollama_embeddings(
            api_base="http://localhost:11434",
            model="test-model",
            prompts=["hello", "world"],
            optional_params={},
            model_response=mock_embedding_response,
            logging_obj=None,
            encoding=mock_encoding,
        )

        assert response.model == "ollama/test-model"
        assert response.object == "list"
        assert isinstance(response.data, list)
        assert response.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_ollama_aembeddings(
    mock_response_data, mock_embedding_response, mock_encoding
):
    mock_response = AsyncMock()
    # Make json() a regular synchronous method, not async
    mock_response.json = MagicMock(return_value=mock_response_data)
    with (
        patch(
            "litellm.module_level_aclient.post", return_value=mock_response
        ) as mock_post,
        patch("litellm.OllamaConfig.get_config", return_value={"truncate": 512}),
    ):

        response = await ollama_aembeddings(
            api_base="http://localhost:11434",
            model="test-model",
            prompts=["hello", "world"],
            optional_params={},
            model_response=mock_embedding_response,
            logging_obj=None,
            encoding=mock_encoding,
        )

        assert response.model == "ollama/test-model"
        assert response.object == "list"
        assert isinstance(response.data, list)
        assert response.usage.total_tokens == 5


def test_prompt_eval_fallback_when_missing(mock_embedding_response, mock_encoding):
    response_data = {
        "embeddings": [[0.1, 0.2, 0.3]],
        # No "prompt_eval_count"
    }

    with (
        patch("litellm.module_level_client.post") as mock_post,
        patch("litellm.OllamaConfig.get_config", return_value={}),
    ):

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_post.return_value = mock_response

        response = ollama_embeddings(
            api_base="http://localhost:11434",
            model="test-model",
            prompts=["only-prompt"],
            optional_params={},
            model_response=mock_embedding_response,
            logging_obj=None,
            encoding=mock_encoding,
        )

        # Fallback should use encoding length (mocked to be 5)
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5
        assert response.usage.completion_tokens == 0
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "api_base",
    [
        "http://gateway.example.com/ollama",
        "http://gateway.example.com/ollama/",
        "http://gateway.example.com/ollama/api/embed/",
    ],
)
@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_ollama_embeddings_ignore_a_trailing_slash_on_api_base(
    api_base: str, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    route: Final = respx.post("http://gateway.example.com/ollama/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 2})
    )

    response: Final = (
        await litellm.aembedding(model="ollama/all-minilm", api_base=api_base, input=["hello", "world"])
        if asynchronous
        else litellm.embedding(model="ollama/all-minilm", api_base=api_base, input=["hello", "world"])
    )

    assert route.call_count == 1
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2], [0.3, 0.4]]


@pytest.fixture
def embedding_route(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> respx.Route:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    return respx_mock.post("http://ollama.test/api/embed").respond(
        200, json={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 2}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("api_key", "env_key", "authorization"),
    [
        ("explicit-key", None, "Bearer explicit-key"),
        (None, "env-key", "Bearer env-key"),
        ("explicit-key", "env-key", "Bearer explicit-key"),
        (None, None, None),
    ],
)
async def test_sdk_embedding_forwards_resolved_auth_and_merged_headers(
    embedding_route: respx.Route,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    api_key: str | None,
    env_key: str | None,
    authorization: str | None,
) -> None:
    if env_key is None:
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_API_KEY", env_key)
    response: Final = (
        await litellm.aembedding(
            model="ollama/nomic-embed-text",
            input="hello",
            api_base="http://ollama.test",
            api_key=api_key,
            headers={"X-Caller": "caller", "X-Shared": "original"},
            extra_headers={"X-Extra": "extra", "X-Shared": "override"},
        )
        if asynchronous
        else litellm.embedding(
            model="ollama/nomic-embed-text",
            input="hello",
            api_base="http://ollama.test",
            api_key=api_key,
            headers={"X-Caller": "caller", "X-Shared": "original"},
            extra_headers={"X-Extra": "extra", "X-Shared": "override"},
        )
    )
    request: Final = embedding_route.calls.last.request

    assert embedding_route.call_count == 1
    assert request.headers.get("Authorization") == authorization
    assert (request.headers["X-Caller"], request.headers["X-Extra"], request.headers["X-Shared"]) == (
        "caller",
        "extra",
        "override",
    )
    assert request.extensions["timeout"] == httpx.Timeout(600).as_dict()
    assert json.loads(request.content) == {"model": "nomic-embed-text", "input": ["hello"]}
    assert isinstance(response, EmbeddingResponse)
    assert response.data[0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("timeout", [0, 2.5, httpx.Timeout(connect=1, read=2, write=3, pool=4)])
async def test_sdk_embedding_applies_timeout_to_http_request(
    embedding_route: respx.Route, asynchronous: bool, timeout: float | httpx.Timeout
) -> None:
    response: Final = (
        await litellm.aembedding(
            model="ollama/nomic-embed-text", input=["hello"], api_base="http://ollama.test", timeout=timeout
        )
        if asynchronous
        else litellm.embedding(
            model="ollama/nomic-embed-text", input=["hello"], api_base="http://ollama.test", timeout=timeout
        )
    )

    assert embedding_route.call_count == 1
    assert embedding_route.calls.last.request.extensions["timeout"] == httpx.Timeout(timeout).as_dict()
    assert isinstance(response, EmbeddingResponse)
    assert response.data[0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("api_key", "headers", "authorization"),
    [
        ("explicit-key", {"X-Custom": "custom"}, "Bearer explicit-key"),
        ("explicit-key", {"Authorization": "Basic caller"}, "Basic caller"),
        ("explicit-key", {"authorization": "Bearer caller"}, "Bearer caller"),
        (None, {}, None),
        ("", {}, None),
    ],
)
@pytest.mark.parametrize("timeout_params", [{}, {"timeout": None}], ids=["omitted", "none"])
async def test_handler_embedding_preserves_caller_headers_and_client_timeout(
    embedding_route: respx.Route,
    asynchronous: bool,
    api_key: str | None,
    headers: Mapping[str, str],
    authorization: str | None,
    timeout_params: Mapping[str, None],
) -> None:
    caller_headers: Final = MappingProxyType(dict(headers))
    expected_timeout: Final = (
        litellm.module_level_aclient.client.timeout if asynchronous else litellm.module_level_client.client.timeout
    )
    response: Final = (
        await ollama_aembeddings(
            "http://ollama.test",
            "nomic-embed-text",
            ["hello"],
            EmbeddingResponse(),
            {},
            None,
            None,
            api_key=api_key,
            headers=caller_headers,
            **timeout_params,
        )
        if asynchronous
        else ollama_embeddings(
            "http://ollama.test",
            "nomic-embed-text",
            ["hello"],
            {},
            EmbeddingResponse(),
            None,
            api_key=api_key,
            headers=caller_headers,
            **timeout_params,
        )
    )
    request: Final = embedding_route.calls.last.request

    assert embedding_route.call_count == 1
    assert request.headers.get_list("Authorization") == ([] if authorization is None else [authorization])
    assert dict(caller_headers) == headers
    assert all(request.headers[name] == value for name, value in caller_headers.items())
    assert request.extensions["timeout"] == expected_timeout.as_dict()
    assert json.loads(request.content) == {"model": "nomic-embed-text", "input": ["hello"]}
    assert response.data[0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sdk_embedding_rejects_non_string_batch(embedding_route: respx.Route, asynchronous: bool) -> None:
    if asynchronous:
        with pytest.raises(litellm.BadRequestError, match="Invalid input for ollama embeddings"):
            await litellm.aembedding(model="ollama/nomic-embed-text", input=[123], api_base="http://ollama.test")
    else:
        with pytest.raises(litellm.BadRequestError, match="Invalid input for ollama embeddings"):
            litellm.embedding(model="ollama/nomic-embed-text", input=[123], api_base="http://ollama.test")

    assert not embedding_route.called
