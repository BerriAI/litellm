"""
Test transformation logic for Docker Model Runner embeddings.

This test verifies that the DockerModelRunnerEmbeddingConfig correctly transforms
embedding requests and handles URL generation for the Docker Model Runner API.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.docker_model_runner.embedding.transformation import (
    DockerModelRunnerEmbeddingConfig,
)


class TestDockerModelRunnerEmbeddingTransformation:
    """Test suite for Docker Model Runner embedding transformation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerEmbeddingConfig()
        self.model = "ai/smollm2"

    def test_get_complete_url_default_api_base(self):
        """Test URL generation with default api_base (no API key required for local DMR)."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/embeddings"

    def test_get_complete_url_with_auto_select_engine(self):
        """Test URL generation with explicit auto-select engine path."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/v1",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/embeddings"

    def test_get_complete_url_with_vllm_engine(self):
        """Test URL generation with explicit vLLM engine path."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/vllm/v1",
            api_key=None,
            model="ai/smollm2-vllm",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/vllm/v1/embeddings"

    def test_get_complete_url_container_host(self):
        """Test URL generation from within a Docker container."""
        url = self.config.get_complete_url(
            api_base="http://model-runner.docker.internal/engines/v1",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert "model-runner.docker.internal" in url
        assert url.endswith("/embeddings")

    def test_get_complete_url_removes_trailing_slash(self):
        """Test that trailing slashes are properly removed from api_base."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/v1/",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/embeddings"
        assert "//v1" not in url

    def test_transform_embedding_request_basic(self):
        """Test basic embedding request transformation with list input."""
        input_data = ["hello world"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        assert result["model"] == self.model
        assert result["input"] == input_data

    def test_transform_embedding_request_string_input(self):
        """Test that string input is converted to list."""
        input_data = "hello world"
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        assert result["input"] == ["hello world"]

    def test_transform_embedding_request_multiple_inputs(self):
        """Test embedding request with multiple input strings."""
        input_data = ["first document", "second document", "third document"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        assert result["input"] == input_data
        assert len(result["input"]) == 3

    def test_transform_embedding_request_with_dimensions(self):
        """Test embedding request with dimensions parameter."""
        input_data = ["hello world"]
        optional_params = {"dimensions": 256}

        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )
        assert result["dimensions"] == 256
        assert result["model"] == self.model
        assert result["input"] == input_data

    def test_transform_embedding_request_with_encoding_format(self):
        """Test embedding request with encoding_format parameter."""
        input_data = ["hello world"]
        optional_params = {"encoding_format": "float"}

        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params=optional_params,
            headers={},
        )
        assert result["encoding_format"] == "float"

    def test_validate_environment_returns_headers(self):
        """Test that validate_environment returns proper headers."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"

    def test_validate_environment_default_api_key(self):
        """Test that validate_environment uses dummy-key when no API key provided."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer dummy-key"

    def test_get_supported_openai_params(self):
        """Test that supported params include standard embedding parameters."""
        supported = self.config.get_supported_openai_params(model=self.model)
        assert "dimensions" in supported
        assert "encoding_format" in supported
        assert "timeout" in supported
        assert "user" in supported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
