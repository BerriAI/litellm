from unittest.mock import MagicMock, patch

import pytest

from litellm import embedding


@pytest.mark.parametrize(
    "set_env, env_value, expected",
    [
        (False, None, "float"),
        (True, "base64", "base64"),
    ],
)
def test_openai_embedding_encoding_format_default(
    monkeypatch, set_env, env_value, expected
):
    monkeypatch.delenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", raising=False)
    if set_env:
        monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", env_value)

    mock_response = MagicMock()
    mock_response.parse.return_value = MagicMock(
        model_dump=lambda: {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-ada-002",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )
    mock_response.headers = {}

    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client"
    ) as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.embeddings.with_raw_response.create.return_value = (
            mock_response
        )

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
        )

        call_kwargs = (
            mock_client_instance.embeddings.with_raw_response.create.call_args[1]
        )
        assert call_kwargs["encoding_format"] == expected


@pytest.mark.parametrize("env_none", ["none", "NONE", " none "])
def test_openai_embedding_encoding_format_env_none_omits_param(
    monkeypatch, env_none
):
    """LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT=none omits encoding_format (provider default)."""
    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", env_none)

    mock_response = MagicMock()
    mock_response.parse.return_value = MagicMock(
        model_dump=lambda: {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-ada-002",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )
    mock_response.headers = {}

    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client"
    ) as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.embeddings.with_raw_response.create.return_value = (
            mock_response
        )

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
        )

        call_kwargs = (
            mock_client_instance.embeddings.with_raw_response.create.call_args[1]
        )
        assert "encoding_format" not in call_kwargs


def test_openai_embedding_encoding_format_explicit_overrides_env(monkeypatch):
    """Request `encoding_format` wins over LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT."""
    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", "float")

    mock_response = MagicMock()
    mock_response.parse.return_value = MagicMock(
        model_dump=lambda: {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-ada-002",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )
    mock_response.headers = {}

    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client"
    ) as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.embeddings.with_raw_response.create.return_value = (
            mock_response
        )

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
            encoding_format="base64",
        )

        call_kwargs = (
            mock_client_instance.embeddings.with_raw_response.create.call_args[1]
        )
        assert call_kwargs["encoding_format"] == "base64"
