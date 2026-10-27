import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.ollama.common_utils import OllamaError
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
def test_ollama_embeddings_ignore_a_trailing_slash_on_api_base(api_base: str) -> None:
    route: Final = respx.post("http://gateway.example.com/ollama/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 2})
    )

    response: Final = litellm.embedding(model="ollama/all-minilm", api_base=api_base, input=["hello", "world"])

    assert route.called
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2], [0.3, 0.4]]


def _non_json_response(status_code: int, body: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = body
    mock_response.headers = {}
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", body, 5)
    return mock_response


def test_ollama_embeddings_non_json_body_raises_ollama_error_instead_of_json_decode_error(
    mock_embedding_response, mock_encoding
):
    with patch("litellm.module_level_client.post") as mock_post:
        mock_post.return_value = _non_json_response(200, "<html><body>sign in</body></html>")

        with pytest.raises(OllamaError) as exc_info:
            ollama_embeddings(
                api_base="http://localhost:11434",
                model="nomic-embed-text",
                prompts=["hello"],
                optional_params={},
                model_response=mock_embedding_response,
                logging_obj=None,
                encoding=mock_encoding,
            )

    assert exc_info.value.status_code == 502
    assert "non-JSON" in exc_info.value.message
    assert "status 200" in exc_info.value.message
    assert "sign in" in exc_info.value.message


@pytest.mark.asyncio
async def test_ollama_aembeddings_non_json_body_keeps_the_upstream_error_status(mock_embedding_response, mock_encoding):
    mock_response = AsyncMock()
    mock_response.json = MagicMock(side_effect=json.JSONDecodeError("Expecting value", "Bad Gateway", 0))
    mock_response.status_code = 502
    mock_response.text = "Bad Gateway"
    mock_response.headers = {}
    with patch("litellm.module_level_aclient.post", return_value=mock_response):
        with pytest.raises(OllamaError) as exc_info:
            await ollama_aembeddings(
                api_base="http://localhost:11434",
                model="nomic-embed-text",
                prompts=["hello"],
                optional_params={},
                model_response=mock_embedding_response,
                logging_obj=None,
                encoding=mock_encoding,
            )

    assert exc_info.value.status_code == 502
    assert "non-JSON" in exc_info.value.message
    assert "Bad Gateway" in exc_info.value.message
