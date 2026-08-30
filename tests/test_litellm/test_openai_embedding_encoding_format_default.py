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
def test_openai_embedding_encoding_format_default(monkeypatch, set_env, env_value, expected):
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

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.embeddings.with_raw_response.create.return_value = mock_response

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
        )

        call_kwargs = mock_client_instance.embeddings.with_raw_response.create.call_args[1]
        assert call_kwargs["encoding_format"] == expected


@pytest.mark.parametrize("env_none", ["none", "NONE", " none "])
def test_openai_embedding_encoding_format_env_none_omits_param(monkeypatch, env_none):
    """LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT=none omits encoding_format from the
    wire request entirely. embeddings.create() hard-defaults the field to "base64"
    whenever the kwarg is absent, so the suppressed path posts the body directly
    (#38661); a chained LiteLLM proxy forwarding to a provider with zero supported
    embedding params (e.g. Bedrock titan) would otherwise reject the injected field."""
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

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.embeddings.with_raw_response.create.return_value = mock_response
        mock_client_instance.post.return_value = MagicMock(
            model_dump=lambda: {
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                "model": "text-embedding-ada-002",
                "object": "list",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        )

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
        )

        # create() cannot represent an absent encoding_format, so the suppressed
        # path must not go through it at all
        mock_client_instance.embeddings.with_raw_response.create.assert_not_called()
        post_args, post_kwargs = mock_client_instance.post.call_args
        assert post_args[0] == "/embeddings"
        assert "encoding_format" not in post_kwargs["body"]


def test_openai_embedding_encoding_format_none_keeps_request_kwargs_out_of_body(monkeypatch):
    """extra_headers (and friends) are embeddings.create() request kwargs, not body
    fields: on the suppressed direct-post path they must be routed into request
    options instead of being serialized into the JSON body."""
    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", "none")

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.post.return_value = MagicMock(
            model_dump=lambda: {
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                "model": "text-embedding-ada-002",
                "object": "list",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        )

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
            extra_headers={"X-Test-Header": "1"},
        )

        mock_client_instance.embeddings.with_raw_response.create.assert_not_called()
        _, post_kwargs = mock_client_instance.post.call_args
        assert "extra_headers" not in post_kwargs["body"], "request kwarg leaked into the JSON body"
        assert post_kwargs["options"]["headers"] == {"X-Test-Header": "1"}
        assert "encoding_format" not in post_kwargs["body"]


def test_openai_aembedding_encoding_format_env_none_omits_param(monkeypatch):
    """Async path: LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT=none must keep
    encoding_format off the wire there too."""
    import asyncio

    from litellm import aembedding

    monkeypatch.setenv("LITELLM_DEFAULT_EMBEDDING_ENCODING_FORMAT", "none")

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        from unittest.mock import AsyncMock

        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.post = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda: {
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                    "model": "text-embedding-ada-002",
                    "object": "list",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
        )

        asyncio.run(aembedding(model="text-embedding-ada-002", input="Hello world"))

        mock_client_instance.embeddings.with_raw_response.create.assert_not_called()
        post_args, post_kwargs = mock_client_instance.post.call_args
        assert post_args[0] == "/embeddings"
        assert "encoding_format" not in post_kwargs["body"]


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

    with patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client") as mock_get_client:
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_client_instance.embeddings.with_raw_response.create.return_value = mock_response

        embedding(
            model="text-embedding-ada-002",
            input="Hello world",
            encoding_format="base64",
        )

        call_kwargs = mock_client_instance.embeddings.with_raw_response.create.call_args[1]
        assert call_kwargs["encoding_format"] == "base64"
