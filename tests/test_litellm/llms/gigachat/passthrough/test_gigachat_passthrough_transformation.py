"""
Unit tests for GigaChatPassthroughConfig transformation.

Tests the GigaChat-specific passthrough configuration including URL construction,
streaming detection, authentication handling, and logging response transformations.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.gigachat.passthrough.transformation import GigaChatPassthroughConfig
from litellm.types.utils import EmbeddingResponse, ModelResponse


def _gigachat_chat_completion_body():
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "GigaChat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from GigaChat",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    }


def _gigachat_embedding_body():
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
                "usage": {"prompt_tokens": 4},
            }
        ],
        "model": "Embeddings",
    }


def _make_httpx_response(body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request(
            "POST", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        ),
    )


class TestGigaChatPassthroughConfig:
    """Tests for GigaChatPassthroughConfig class."""

    def test_is_streaming_request_true(self):
        """Test streaming is detected when stream=True."""
        config = GigaChatPassthroughConfig()
        assert (
            config.is_streaming_request("chat/completions", {"stream": True}) is True
        )

    def test_is_streaming_request_false(self):
        """Test streaming is not detected when stream=False."""
        config = GigaChatPassthroughConfig()
        assert (
            config.is_streaming_request("chat/completions", {"stream": False})
            is False
        )

    def test_is_streaming_request_missing_stream_key(self):
        """Test streaming defaults to False when stream key is missing."""
        config = GigaChatPassthroughConfig()
        assert (
            config.is_streaming_request("chat/completions", {"model": "GigaChat"})
            is False
        )

    def test_get_complete_url_with_api_base(self):
        """Test URL construction with explicit api_base."""
        config = GigaChatPassthroughConfig()
        api_base = "https://custom.gigachat.ru/api/v1"
        endpoint = "chat/completions"

        complete_url, base_target_url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="GigaChat",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url) == f"{api_base}/{endpoint}"
        assert base_target_url == api_base

    def test_get_complete_url_with_leading_slash_endpoint(self):
        """Test URL construction with endpoint having leading slash."""
        config = GigaChatPassthroughConfig()
        api_base = "https://custom.gigachat.ru/api/v1"
        endpoint = "/chat/completions"

        complete_url, base_target_url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="GigaChat",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={},
        )

        assert str(complete_url) == "https://custom.gigachat.ru/api/v1/chat/completions"
        assert base_target_url == api_base

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_complete_url_with_env_api_base(self, mock_get_secret):
        """Test URL construction with api_base from environment."""
        config = GigaChatPassthroughConfig()
        env_api_base = "https://env.gigachat.ru/api/v1"
        mock_get_secret.return_value = env_api_base

        complete_url, base_target_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="GigaChat",
            endpoint="embeddings",
            request_query_params=None,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url).startswith(env_api_base)
        assert base_target_url == env_api_base
        mock_get_secret.assert_called_once_with("GIGACHAT_API_BASE")

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_complete_url_fallback_to_default(self, mock_get_secret):
        """Test URL construction falls back to default GIGACHAT_BASE_URL."""
        config = GigaChatPassthroughConfig()
        mock_get_secret.return_value = None

        complete_url, base_target_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="GigaChat",
            endpoint="models",
            request_query_params=None,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert "gigachat.devices.sberbank.ru" in str(complete_url)
        assert base_target_url == "https://gigachat.devices.sberbank.ru/api/v1"

    def test_get_complete_url_no_api_base_raises(self):
        """Test that exception is raised when no api_base can be resolved."""
        config = GigaChatPassthroughConfig()
        with patch(
            "litellm.llms.gigachat.passthrough.transformation.get_secret_str",
            return_value=None,
        ):
            with patch(
                "litellm.llms.gigachat.passthrough.transformation.GIGACHAT_BASE_URL",
                None,
            ):
                with pytest.raises(Exception, match="GigaChat api base not found"):
                    config.get_complete_url(
                        api_base=None,
                        api_key=None,
                        model="GigaChat",
                        endpoint="chat/completions",
                        request_query_params=None,
                        litellm_params={},
                    )

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_access_token"
    )
    def test_validate_environment(self, mock_get_access_token):
        """Test headers are set correctly with OAuth token."""
        config = GigaChatPassthroughConfig()
        mock_get_access_token.return_value = "test-token-123"

        headers = config.validate_environment(
            headers={},
            model="GigaChat",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="test-credentials",
            api_base="https://custom.gigachat.ru",
        )

        assert headers["Authorization"] == "Bearer test-token-123"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        mock_get_access_token.assert_called_once_with(
            credentials="test-credentials",
            litellm_params={},
        )

    def test_logging_non_streaming_response_chat_completions(self):
        """Test chat completions endpoint returns ModelResponse."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        result = config.logging_non_streaming_response(
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            httpx_response=_make_httpx_response(_gigachat_chat_completion_body()),
            request_data={
                "model": "gigachat/GigaChat",
                "messages": [{"role": "user", "content": "hi"}],
            },
            logging_obj=logging_obj,
            endpoint="chat/completions",
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Hello from GigaChat"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3
        assert result.usage.total_tokens == 8

    def test_logging_non_streaming_response_embeddings(self):
        """Test embeddings endpoint returns EmbeddingResponse."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        result = config.logging_non_streaming_response(
            model="gigachat/Embeddings",
            custom_llm_provider="gigachat",
            httpx_response=_make_httpx_response(_gigachat_embedding_body()),
            request_data={"input": ["hello"], "model": "gigachat/Embeddings"},
            logging_obj=logging_obj,
            endpoint="embeddings",
        )

        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 1
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]

    def test_logging_non_streaming_response_unknown_endpoint_returns_none(self):
        """Test unknown endpoint returns None."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        result = config.logging_non_streaming_response(
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            httpx_response=_make_httpx_response(_gigachat_chat_completion_body()),
            request_data={},
            logging_obj=logging_obj,
            endpoint="images/generations",
        )

        assert result is None

    def test_handle_logging_collected_chunks_with_string_chunks(self):
        """Test converting string chunks to model response."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        chunks = [
            '{"choices": [{"delta": {"content": "Hello"}, "index": 0}]}',
            '{"choices": [{"delta": {"content": " world"}, "index": 0}]}',
            '{"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}',
        ]

        result = config.handle_logging_collected_chunks(
            all_chunks=chunks,
            litellm_logging_obj=logging_obj,
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            endpoint="chat/completions",
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Hello world"

    def test_handle_logging_collected_chunks_with_bytes_chunks(self):
        """Test converting bytes chunks to model response."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        chunks = [
            b'{"choices": [{"delta": {"content": "Hi"}, "index": 0}]}',
            b'{"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}',
        ]

        result = config.handle_logging_collected_chunks(
            all_chunks=chunks,
            litellm_logging_obj=logging_obj,
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            endpoint="chat/completions",
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Hi"

    def test_handle_logging_collected_chunks_with_done_and_empty(self):
        """Test that [DONE] and empty chunks are skipped."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        chunks = [
            "",
            "[DONE]",
            '{"choices": [{"delta": {"content": "test"}, "index": 0}]}',
            '{"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}',
        ]

        result = config.handle_logging_collected_chunks(
            all_chunks=chunks,
            litellm_logging_obj=logging_obj,
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            endpoint="chat/completions",
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "test"

    def test_handle_logging_collected_chunks_with_dict_chunks(self):
        """Test converting dict chunks directly."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        chunks = [
            {"choices": [{"delta": {"content": "direct"}, "index": 0}]},
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ]

        result = config.handle_logging_collected_chunks(
            all_chunks=chunks,
            litellm_logging_obj=logging_obj,
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            endpoint="chat/completions",
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "direct"

    def test_handle_logging_collected_chunks_empty_list_returns_none(self):
        """Test empty chunks list returns None."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        result = config.handle_logging_collected_chunks(
            all_chunks=[],
            litellm_logging_obj=logging_obj,
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            endpoint="chat/completions",
        )

        assert result is None

    def test_handle_logging_collected_chunks_invalid_json_skipped(self):
        """Test invalid JSON chunks are skipped gracefully."""
        config = GigaChatPassthroughConfig()
        logging_obj = MagicMock()

        chunks = [
            "not-valid-json",
            '{"choices": [{"delta": {"content": "valid"}, "index": 0}]}',
            '{"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}',
        ]

        result = config.handle_logging_collected_chunks(
            all_chunks=chunks,
            litellm_logging_obj=logging_obj,
            model="gigachat/GigaChat",
            custom_llm_provider="gigachat",
            endpoint="chat/completions",
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "valid"

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_api_base_with_explicit_value(self, mock_get_secret):
        """Test get_api_base returns explicit value when provided."""
        explicit_base = "https://custom.gigachat.ru/api/v1"
        result = GigaChatPassthroughConfig.get_api_base(api_base=explicit_base)
        assert result == explicit_base
        mock_get_secret.assert_not_called()

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_api_base_from_environment(self, mock_get_secret):
        """Test get_api_base retrieves from environment when not provided."""
        env_base = "https://env.gigachat.ru/api/v1"
        mock_get_secret.return_value = env_base
        result = GigaChatPassthroughConfig.get_api_base(api_base=None)
        assert result == env_base
        mock_get_secret.assert_called_once_with("GIGACHAT_API_BASE")

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_api_base_fallback_to_default(self, mock_get_secret):
        """Test get_api_base falls back to GIGACHAT_BASE_URL."""
        mock_get_secret.return_value = None
        result = GigaChatPassthroughConfig.get_api_base(api_base=None)
        assert result == "https://gigachat.devices.sberbank.ru/api/v1"

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_api_key_with_explicit_value(self, mock_get_secret):
        """Test get_api_key returns explicit value when provided."""
        explicit_key = "test-api-key"
        result = GigaChatPassthroughConfig.get_api_key(api_key=explicit_key)
        assert result == explicit_key
        mock_get_secret.assert_not_called()

    @patch(
        "litellm.llms.gigachat.passthrough.transformation.get_secret_str"
    )
    def test_get_api_key_from_environment(self, mock_get_secret):
        """Test get_api_key retrieves from environment when not provided."""
        env_key = "env-api-key"
        mock_get_secret.return_value = env_key
        result = GigaChatPassthroughConfig.get_api_key(api_key=None)
        assert result == env_key
        mock_get_secret.assert_called_once_with("GIGACHAT_API_KEY")

    def test_get_base_model_returns_model(self):
        """Test get_base_model returns the model as-is."""
        model = "gigachat/GigaChat"
        result = GigaChatPassthroughConfig.get_base_model(model)
        assert result == model

    def test_get_models(self):
        """Test get_models delegates to base class."""
        config = GigaChatPassthroughConfig()
        result = config.get_models()
        assert result == []
