from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.ollama.completion.handler import (
    _prepare_ollama_embedding_payload,
    ollama_aembeddings,
    ollama_embeddings,
)
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
    with patch("litellm.module_level_client.post") as mock_post, patch(
        "litellm.OllamaConfig.get_config", return_value={"truncate": 512}
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


def test_prepare_ollama_embedding_payload_strips_ollama_prefix():
    payload = _prepare_ollama_embedding_payload(
        model="ollama/nomic-embed-text",
        prompts=["hello"],
        optional_params={},
    )

    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == ["hello"]


def test_prepare_ollama_embedding_payload_keeps_unprefixed_model():
    payload = _prepare_ollama_embedding_payload(
        model="nomic-embed-text",
        prompts=["hello"],
        optional_params={},
    )

    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == ["hello"]


def test_ollama_embeddings_strips_prefix_in_embed_payload(
    mock_response_data, mock_embedding_response, mock_encoding
):
    with patch("litellm.module_level_client.post") as mock_post, patch(
        "litellm.OllamaConfig.get_config", return_value={"truncate": 512}
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response

        ollama_embeddings(
            api_base="http://localhost:11434",
            model="ollama/nomic-embed-text",
            prompts=["hello", "world"],
            optional_params={},
            model_response=mock_embedding_response,
            logging_obj=None,
            encoding=mock_encoding,
        )

        assert mock_post.call_args.kwargs["json"]["model"] == "nomic-embed-text"
        assert mock_post.call_args.kwargs["json"]["input"] == ["hello", "world"]


@pytest.mark.asyncio
async def test_ollama_aembeddings(
    mock_response_data, mock_embedding_response, mock_encoding
):
    mock_response = AsyncMock()
    # Make json() a regular synchronous method, not async
    mock_response.json = MagicMock(return_value=mock_response_data)
    with patch(
        "litellm.module_level_aclient.post", return_value=mock_response
    ) as mock_post, patch(
        "litellm.OllamaConfig.get_config", return_value={"truncate": 512}
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


@pytest.mark.asyncio
async def test_ollama_aembeddings_strips_prefix_in_embed_payload(
    mock_response_data, mock_embedding_response, mock_encoding
):
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=mock_response_data)
    with patch(
        "litellm.module_level_aclient.post", return_value=mock_response
    ) as mock_post, patch(
        "litellm.OllamaConfig.get_config", return_value={"truncate": 512}
    ):
        await ollama_aembeddings(
            api_base="http://localhost:11434",
            model="ollama/nomic-embed-text",
            prompts=["hello", "world"],
            optional_params={},
            model_response=mock_embedding_response,
            logging_obj=None,
            encoding=mock_encoding,
        )

        assert mock_post.call_args.kwargs["json"]["model"] == "nomic-embed-text"
        assert mock_post.call_args.kwargs["json"]["input"] == ["hello", "world"]


def test_prompt_eval_fallback_when_missing(mock_embedding_response, mock_encoding):
    response_data = {
        "embeddings": [[0.1, 0.2, 0.3]],
        # No "prompt_eval_count"
    }

    with patch("litellm.module_level_client.post") as mock_post, patch(
        "litellm.OllamaConfig.get_config", return_value={}
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
