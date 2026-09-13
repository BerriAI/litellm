"""
Unit tests for GigaChat embedding transformation.

Tests GigaChatEmbeddingConfig covering get_config, get_supported_openai_params,
map_openai_params, _get_openai_compatible_provider_info, get_complete_url,
transform_embedding_request, transform_embedding_response, validate_environment,
and get_error_class.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm import LlmProviders
from litellm.llms.gigachat.embedding.transformation import (
    GigaChatEmbeddingConfig,
    GigaChatEmbeddingError,
)
from litellm.types.utils import EmbeddingResponse

TRANSFORM_MODULE = "litellm.llms.gigachat.embedding.transformation"


def _make_httpx_response(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", "https://gigachat.devices.sberbank.ru/api/v1/embeddings"),
    )


# ---------------------------------------------------------------------------
# GigaChatEmbeddingConfig
# ---------------------------------------------------------------------------


class TestGetConfig:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_contains_only_abc_impl(self):
        """get_config returns ABC internal data due to inheritance."""
        result = self.config.get_config()
        # The only key should be _abc_impl from ABC base class
        assert set(result.keys()) == {"_abc_impl"}


class TestGetSupportedOpenAiParams:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_returns_empty_list(self):
        params = self.config.get_supported_openai_params("GigaChat")
        assert params == []


class TestMapOpenAiParams:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_returns_optional_params_unchanged(self):
        result = self.config.map_openai_params(
            non_default_params={"model": "test"},
            optional_params={"temperature": 0.5},
            model="GigaChat",
            drop_params=False,
        )
        assert result == {"temperature": 0.5}

    def test_returns_empty_dict_when_no_optional_params(self):
        result = self.config.map_openai_params(
            non_default_params={},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result == {}


class TestGetOpenaiCompatibleProviderInfo:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_returns_gigachat_provider(self):
        provider, api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base="https://api.example.com", api_key="test-key"
        )
        assert provider == LlmProviders.GIGACHAT.value
        assert api_base == "https://api.example.com"
        assert api_key == "test-key"

    def test_resolves_api_base_when_none(self, monkeypatch):
        monkeypatch.delenv("GIGACHAT_API_BASE", raising=False)
        provider, api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key="key"
        )
        assert api_base is not None
        assert api_base.endswith("/api/v1")

    def test_returns_none_api_key(self):
        _, _, api_key = self.config._get_openai_compatible_provider_info(
            api_base="https://example.com", api_key=None
        )
        assert api_key is None


class TestGetCompleteUrl:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_default_url(self):
        url = self.config.get_complete_url(
            api_base=None, api_key=None, model="GigaChat",
            optional_params={}, litellm_params={},
        )
        assert url.endswith("/embeddings")

    def test_custom_api_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.example.com", api_key=None, model="GigaChat",
            optional_params={}, litellm_params={},
        )
        assert url == "https://custom.example.com/embeddings"

    def test_trailing_slash_api_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.example.com/", api_key=None, model="GigaChat",
            optional_params={}, litellm_params={},
        )
        # get_api_base doesn't strip slash, so we get double slash
        assert url == "https://custom.example.com//embeddings"


class TestTransformEmbeddingRequest:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_string_input(self):
        result = self.config.transform_embedding_request(
            model="gigachat/Embeddings",
            input="hello world",
            optional_params={},
            headers={},
        )
        assert result == {"model": "Embeddings", "input": ["hello world"]}

    def test_list_input(self):
        result = self.config.transform_embedding_request(
            model="gigachat/Embeddings",
            input=["text1", "text2"],
            optional_params={},
            headers={},
        )
        assert result == {"model": "Embeddings", "input": ["text1", "text2"]}

    def test_strips_gigachat_prefix(self):
        result = self.config.transform_embedding_request(
            model="gigachat/GigaChat-Pro",
            input="test",
            optional_params={},
            headers={},
        )
        assert result["model"] == "GigaChat-Pro"

    def test_model_without_prefix(self):
        result = self.config.transform_embedding_request(
            model="Embeddings",
            input="test",
            optional_params={},
            headers={},
        )
        assert result["model"] == "Embeddings"


class TestTransformEmbeddingResponse:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()
        self.logging_obj = MagicMock()

    def _make_gigachat_response(self, data: list[dict]) -> httpx.Response:
        return _make_httpx_response({
            "object": "list",
            "data": data,
            "model": "Embeddings",
        })

    def test_basic_response(self):
        raw = self._make_gigachat_response([
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ])
        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="gigachat/Embeddings",
            raw_response=raw,
            model_response=model_response,
            logging_obj=self.logging_obj,
            api_key="test-key",
            request_data={"input": ["text"]},
            optional_params={},
            litellm_params={},
        )
        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[0]["index"] == 0
        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    def test_aggregates_per_embedding_usage(self):
        raw = self._make_gigachat_response([
            {
                "object": "embedding",
                "embedding": [0.1, 0.2],
                "index": 0,
                "usage": {"prompt_tokens": 5},
            },
            {
                "object": "embedding",
                "embedding": [0.3, 0.4],
                "index": 1,
                "usage": {"prompt_tokens": 7},
            },
        ])
        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="gigachat/Embeddings",
            raw_response=raw,
            model_response=model_response,
            logging_obj=self.logging_obj,
            api_key="test-key",
            request_data={"input": ["a", "b"]},
            optional_params={},
            litellm_params={},
        )
        # Total should be sum of per-embedding prompt_tokens
        assert result.usage.prompt_tokens == 12
        assert result.usage.total_tokens == 12
        # Usage should be removed from individual embedding data
        assert "usage" not in result.data[0]
        assert "usage" not in result.data[1]

    def test_usage_removed_from_individual_embeddings(self):
        raw = self._make_gigachat_response([
            {
                "object": "embedding",
                "embedding": [0.5],
                "index": 0,
                "usage": {"prompt_tokens": 3},
            }
        ])
        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="gigachat/Embeddings",
            raw_response=raw,
            model_response=model_response,
            logging_obj=self.logging_obj,
            api_key="key",
            request_data={"input": ["x"]},
            optional_params={},
            litellm_params={},
        )
        # usage should NOT be in the final EmbeddingResponse data items
        for emb in result.data:
            assert "usage" not in emb

    def test_passes_model_from_response(self):
        raw = self._make_gigachat_response([
            {"object": "embedding", "embedding": [0.1], "index": 0},
        ])
        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="gigachat/Embeddings",
            raw_response=raw,
            model_response=model_response,
            logging_obj=self.logging_obj,
            api_key="key",
            request_data={"input": ["x"]},
            optional_params={},
            litellm_params={},
        )
        assert result.model == "Embeddings"

    def test_calls_logging_post_call(self):
        raw = self._make_gigachat_response([
            {"object": "embedding", "embedding": [0.1], "index": 0},
        ])
        model_response = EmbeddingResponse()
        self.config.transform_embedding_response(
            model="gigachat/Embeddings",
            raw_response=raw,
            model_response=model_response,
            logging_obj=self.logging_obj,
            api_key="test-api-key",
            request_data={"input": ["hello"]},
            optional_params={},
            litellm_params={},
        )
        self.logging_obj.post_call.assert_called_once()
        args = self.logging_obj.post_call.call_args.kwargs
        assert args["api_key"] == "test-api-key"
        assert args["input"] == ["hello"]


class TestValidateEnvironment:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    @patch(f"{TRANSFORM_MODULE}.get_access_token", return_value="test-token")
    def test_sets_oauth_headers(self, mock_get_token):
        headers = self.config.validate_environment(
            headers={},
            model="GigaChat",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="creds",
            api_base="https://api.example.com",
        )
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"
        mock_get_token.assert_called_once_with(credentials="creds", litellm_params={})

    @patch(f"{TRANSFORM_MODULE}.get_access_token", return_value="token")
    def test_merges_custom_headers(self, mock_get_token):
        headers = self.config.validate_environment(
            headers={"X-Custom": "value"},
            model="GigaChat",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="creds",
            api_base="https://api.example.com",
        )
        assert headers["Authorization"] == "Bearer token"
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom"] == "value"

    @patch(f"{TRANSFORM_MODULE}.get_access_token", return_value="token")
    def test_custom_header_overwrites_default(self, mock_get_token):
        headers = self.config.validate_environment(
            headers={"Authorization": "Bearer custom"},
            model="GigaChat",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="creds",
            api_base="https://api.example.com",
        )
        # Merge: default headers first, then custom headers on top
        assert headers["Authorization"] == "Bearer custom"


class TestGetErrorClass:
    def setup_method(self):
        self.config = GigaChatEmbeddingConfig()

    def test_returns_gigachat_embedding_error(self):
        error = self.config.get_error_class(
            error_message="embedding failed",
            status_code=400,
            headers={"x-request-id": "abc"},
        )
        assert isinstance(error, GigaChatEmbeddingError)
        assert error.status_code == 400
        assert error.message == "embedding failed"