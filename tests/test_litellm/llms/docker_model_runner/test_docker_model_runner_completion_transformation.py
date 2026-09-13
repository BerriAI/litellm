"""
Test transformation logic for Docker Model Runner text completions.

This test verifies that the DockerModelRunnerCompletionConfig correctly transforms
text completion requests and handles URL generation for the Docker Model Runner API.
"""

import pytest

from litellm.llms.docker_model_runner.completion.transformation import (
    DockerModelRunnerCompletionConfig,
)


class TestDockerModelRunnerCompletionTransformation:
    """Test suite for Docker Model Runner text completion transformation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerCompletionConfig()
        self.model = "ai/smollm2"

    def test_get_complete_url_default_api_base(self):
        """Test URL generation with default api_base."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/v1/completions"

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
        assert url == "http://localhost:12434/engines/v1/completions"

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
        assert url == "http://localhost:12434/engines/vllm/v1/completions"

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
        assert url.endswith("/completions")

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
        assert url == "http://localhost:12434/engines/v1/completions"
        assert "//v1" not in url

    def test_transform_text_completion_request_basic(self):
        """Test basic text completion request transformation."""
        result = self.config.transform_text_completion_request(
            model=self.model,
            messages=[{"role": "user", "content": "Once upon a time"}],
            optional_params={},
            headers={},
        )
        assert result["model"] == self.model
        assert result["prompt"] == "Once upon a time"

    def test_transform_text_completion_request_with_max_tokens(self):
        """Test completion request with max_tokens parameter."""
        optional_params = {"max_tokens": 100, "temperature": 0.7}

        result = self.config.transform_text_completion_request(
            model=self.model,
            messages=[{"role": "user", "content": "Once upon a time"}],
            optional_params=optional_params,
            headers={},
        )
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7

    def test_transform_text_completion_request_stop_sequences(self):
        """Test completion request with stop sequences."""
        optional_params = {"stop": ["\n\n", "END"]}

        result = self.config.transform_text_completion_request(
            model=self.model,
            messages=[{"role": "user", "content": "Once upon a time"}],
            optional_params=optional_params,
            headers={},
        )
        assert result["stop"] == ["\n\n", "END"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
