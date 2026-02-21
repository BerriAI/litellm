"""
Unit tests for GDM Embedding configuration.

GDM (https://ai.gdm.se) provides an OpenAI-compatible API.
"""

import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.gdm.embed.transformation import GDMEmbeddingConfig, GDMEmbeddingError


class TestGDMEmbeddingConfig:
    """Test class for GDM Embedding functionality"""

    @pytest.fixture
    def config(self):
        """Get GDM embedding config"""
        return GDMEmbeddingConfig()

    def test_get_supported_openai_params(self, config):
        """
        Test that get_supported_openai_params returns expected params
        """
        supported_params = config.get_supported_openai_params(model="text-embedding-ada-002")

        # Check common embedding params are supported
        assert "encoding_format" in supported_params
        assert "dimensions" in supported_params
        assert "user" in supported_params

    def test_validate_environment_with_api_key(self, config):
        """
        Test that validate_environment returns proper headers with API key
        """
        headers = config.validate_environment(
            headers={},
            model="text-embedding-ada-002",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_custom_headers_merge(self, config):
        """
        Test that custom headers are merged with defaults
        """
        custom_headers = {"X-Custom-Header": "custom-value"}

        headers = config.validate_environment(
            headers=custom_headers,
            model="text-embedding-ada-002",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["X-Custom-Header"] == "custom-value"

    def test_get_complete_url_default(self, config):
        """
        Test that get_complete_url constructs the correct endpoint URL
        """
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://ai.gdm.se/api/v1/embeddings"

    def test_get_complete_url_with_custom_base(self, config):
        """
        Test that get_complete_url works with custom api_base
        """
        url = config.get_complete_url(
            api_base="https://custom.gdm.se/api/v1",
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://custom.gdm.se/api/v1/embeddings"

    def test_get_complete_url_strips_trailing_slash(self, config):
        """
        Test that get_complete_url properly handles trailing slashes
        """
        url = config.get_complete_url(
            api_base="https://custom.gdm.se/api/v1/",
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://custom.gdm.se/api/v1/embeddings"

    def test_get_complete_url_from_env(self, config, monkeypatch):
        """
        Test that get_complete_url reads from environment
        """
        monkeypatch.setenv("GDM_API_BASE", "https://env.gdm.se/api/v1")

        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://env.gdm.se/api/v1/embeddings"

    def test_map_openai_params_encoding_format(self, config):
        """
        Test that encoding_format parameter is correctly mapped
        """
        non_default_params = {"encoding_format": "float"}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="text-embedding-ada-002",
            drop_params=False
        )

        assert result.get("encoding_format") == "float"

    def test_map_openai_params_dimensions(self, config):
        """
        Test that dimensions parameter is correctly mapped
        """
        non_default_params = {"dimensions": 512}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="text-embedding-ada-002",
            drop_params=False
        )

        assert result.get("dimensions") == 512

    def test_map_openai_params_multiple(self, config):
        """
        Test that multiple parameters are correctly mapped
        """
        non_default_params = {
            "encoding_format": "base64",
            "dimensions": 256,
            "user": "test-user"
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="text-embedding-ada-002",
            drop_params=False
        )

        assert result.get("encoding_format") == "base64"
        assert result.get("dimensions") == 256
        assert result.get("user") == "test-user"

    def test_map_openai_params_unsupported_dropped(self, config):
        """
        Test that unsupported parameters are not mapped
        """
        non_default_params = {
            "dimensions": 512,
            "unsupported_param": "should_be_dropped"
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="text-embedding-ada-002",
            drop_params=False
        )

        assert result.get("dimensions") == 512
        assert "unsupported_param" not in result

    def test_transform_embedding_request_string_input(self, config):
        """
        Test that transform_embedding_request handles string input
        """
        request = config.transform_embedding_request(
            model="text-embedding-ada-002",
            input="Hello, world!",
            optional_params={},
            headers={}
        )

        assert request["model"] == "text-embedding-ada-002"
        assert request["input"] == ["Hello, world!"]

    def test_transform_embedding_request_list_input(self, config):
        """
        Test that transform_embedding_request handles list input
        """
        request = config.transform_embedding_request(
            model="text-embedding-ada-002",
            input=["Hello", "World"],
            optional_params={},
            headers={}
        )

        assert request["model"] == "text-embedding-ada-002"
        assert request["input"] == ["Hello", "World"]

    def test_transform_embedding_request_strips_prefix(self, config):
        """
        Test that transform_embedding_request strips 'gdm/' prefix from model
        """
        request = config.transform_embedding_request(
            model="gdm/text-embedding-ada-002",
            input="Hello",
            optional_params={},
            headers={}
        )

        assert request["model"] == "text-embedding-ada-002"

    def test_transform_embedding_request_with_optional_params(self, config):
        """
        Test that transform_embedding_request includes optional params
        """
        request = config.transform_embedding_request(
            model="text-embedding-ada-002",
            input="Hello",
            optional_params={"dimensions": 256, "encoding_format": "float"},
            headers={}
        )

        assert request["model"] == "text-embedding-ada-002"
        assert request["dimensions"] == 256
        assert request["encoding_format"] == "float"

    def test_get_error_class(self, config):
        """
        Test that get_error_class returns GDMEmbeddingError
        """
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )

        assert isinstance(error, GDMEmbeddingError)
        assert error.status_code == 400
        assert error.message == "Test error"


class TestGDMEmbeddingError:
    """Test class for GDMEmbeddingError"""

    def test_gdm_embedding_error_creation(self):
        """
        Test that GDMEmbeddingError can be created with required parameters
        """
        exception = GDMEmbeddingError(
            status_code=400,
            message="Bad Request"
        )

        assert exception.status_code == 400
        assert exception.message == "Bad Request"

    def test_gdm_embedding_error_with_headers(self):
        """
        Test that GDMEmbeddingError can be created with headers
        """
        headers = {"Content-Type": "application/json", "X-Request-Id": "12345"}
        exception = GDMEmbeddingError(
            status_code=500,
            message="Internal Server Error",
            headers=headers
        )

        assert exception.status_code == 500
        assert exception.message == "Internal Server Error"
        assert exception.headers == headers
