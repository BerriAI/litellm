import base64
import struct
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
async def test_ollama_aembeddings(mock_response_data, mock_embedding_response, mock_encoding):
    mock_response = AsyncMock()
    # Make json() a regular synchronous method, not async
    mock_response.json = MagicMock(return_value=mock_response_data)
    with (
        patch("litellm.module_level_aclient.post", return_value=mock_response),
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


def _embed(returned_vector: list[float], **kwargs) -> tuple[EmbeddingResponse, dict]:
    """Goes through the public entrypoint on purpose: param mapping used to drop
    ``encoding_format`` before the handler saw it, so handler-level tests proved nothing."""
    with patch("litellm.module_level_client.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [returned_vector],
            "prompt_eval_count": 3,
        }
        mock_post.return_value = mock_response

        response = litellm.embedding(
            model="ollama/nomic-embed-text",
            input=["hello world"],
            api_base="http://localhost:11434",
            **kwargs,
        )

        # litellm also POSTs to /api/show for model metadata, and that call lands last.
        # Select the embed request explicitly so payload assertions cannot pass vacuously.
        embed_payloads: Final = [
            call.kwargs["json"]
            for call in mock_post.call_args_list
            if call.kwargs.get("url", "").endswith("/api/embed")
        ]
        assert len(embed_payloads) == 1

    return response, embed_payloads[0]


@pytest.mark.parametrize("encoding_format", ["float", "base64"])
def test_encoding_format_is_accepted_without_drop_params(encoding_format):
    """Used to raise ``UnsupportedParamsError`` for ollama unless the caller opted into
    ``drop_params``, reporting a standard OpenAI param as a server error."""
    response, _ = _embed([0.1, 0.2, 0.3], encoding_format=encoding_format)

    assert response.data[0]["embedding"] is not None


def test_base64_request_returns_decodable_float32():
    """Regression test for the 768 -> 192 truncation: litellm returned a raw float list
    whatever the format, and the openai SDK decoded each float as one byte."""
    vector: Final = [i / 1000 for i in range(768)]

    response, _ = _embed(vector, encoding_format="base64")

    encoded = response.data[0]["embedding"]
    assert isinstance(encoded, str)

    raw: Final = base64.b64decode(encoded)
    decoded: Final = struct.unpack(f"<{len(raw) // 4}f", raw)

    assert len(decoded) == len(vector)
    assert decoded == pytest.approx(vector, abs=1e-6)


def test_float_request_returns_plain_list():
    vector: Final = [0.1, 0.2, 0.3]

    response, _ = _embed(vector, encoding_format="float")

    assert response.data[0]["embedding"] == pytest.approx(vector)


def test_omitted_encoding_format_still_returns_plain_list():
    vector: Final = [0.1, 0.2, 0.3]

    response, _ = _embed(vector)

    assert response.data[0]["embedding"] == pytest.approx(vector)


@pytest.mark.parametrize("encoding_format", ["float", "base64"])
def test_encoding_format_is_not_sent_to_ollama(encoding_format):
    """Forwarding it would land in Ollama's ``options`` dict, which sets model runtime
    options rather than selecting a response encoding."""
    _, sent = _embed([0.1, 0.2, 0.3], encoding_format=encoding_format)

    assert "encoding_format" not in sent
    assert "encoding_format" not in sent.get("options", {})


def test_dimensions_mismatch_raises_instead_of_returning_wrong_width():
    """Callers used to get a 200 and find the wrong width downstream. A width the model
    cannot produce is a caller error, so it must be 400-class, not an internal error."""
    with pytest.raises(litellm.BadRequestError) as exc_info:
        _embed([0.1, 0.2, 0.3], dimensions=768)

    message: Final = str(exc_info.value)
    assert "768" in message
    assert "3" in message


def test_dimensions_match_is_accepted():
    vector: Final = [0.1, 0.2, 0.3]

    response, _ = _embed(vector, dimensions=3)

    assert response.data[0]["embedding"] == pytest.approx(vector)


def test_base64_and_dimensions_together_validate_true_width():
    """The width check must run against the real vector, not the encoded string."""
    vector: Final = [i / 1000 for i in range(768)]

    response, _ = _embed(vector, encoding_format="base64", dimensions=768)

    raw: Final = base64.b64decode(response.data[0]["embedding"])
    assert len(raw) // 4 == 768
