"""
Test transformation logic for Docker Model Runner image generation.

This test verifies that the DockerModelRunnerImageGenerationConfig correctly
transforms image generation requests and handles URL generation for the
Diffusers backend of Docker Model Runner.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.docker_model_runner.images.transformation import (
    DockerModelRunnerImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestDockerModelRunnerImageGenerationTransformation:
    """Test suite for Docker Model Runner image generation transformation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerImageGenerationConfig()
        self.model = "stable-diffusion:Q4"

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
        assert url == "http://localhost:12434/engines/diffusers/v1/images/generations"

    def test_get_complete_url_with_host_only(self):
        """Test URL generation with host-only api_base."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/diffusers/v1/images/generations"

    def test_get_complete_url_container_host(self):
        """Test URL generation from within a Docker container."""
        url = self.config.get_complete_url(
            api_base="http://model-runner.docker.internal",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert "model-runner.docker.internal" in url
        assert url.endswith("/engines/diffusers/v1/images/generations")

    def test_get_complete_url_custom_port(self):
        """Test URL generation with custom TCP port."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12435",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12435/engines/diffusers/v1/images/generations"

    def test_get_complete_url_with_trailing_slash(self):
        """Test that trailing slashes are properly removed from api_base."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/",
            api_key=None,
            model=self.model,
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "http://localhost:12434/engines/diffusers/v1/images/generations"
        assert "//" not in url.replace("http://", "")

    def test_transform_image_generation_request_basic(self):
        """Test basic image generation request transformation."""
        prompt = "A picture of a nice cat"
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["model"] == self.model
        assert result["prompt"] == prompt

    def test_transform_image_generation_request_with_size(self):
        """Test image generation request with size parameter."""
        prompt = "A picture of a nice cat"
        optional_params = {"size": "512x512"}

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert result["size"] == "512x512"

    def test_transform_image_generation_request_multiple_params(self):
        """Test image generation request with multiple parameters."""
        prompt = "A sunset over the ocean"
        optional_params = {"size": "1024x768", "n": 2, "response_format": "b64_json"}

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert result["model"] == self.model
        assert result["prompt"] == prompt
        assert result["size"] == "1024x768"
        assert result["n"] == 2
        assert result["response_format"] == "b64_json"

    def test_transform_image_generation_response_single_image(self):
        """Test image generation response with a single image."""
        # Minimal valid PNG in base64
        b64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        response_data = {
            "data": [
                {"b64_json": b64_image},
            ]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)

        model_response = ImageResponse()

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == b64_image
        assert result.data[0].url is None
        assert result.data[0].revised_prompt is None

    def test_transform_image_generation_response_multiple_images(self):
        """Test image generation response with multiple images."""
        b64_image_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        b64_image_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNgAAIAAgEBAAIA9v9/AAAVFJREFUeJxjYGRgYOBgAOJk//9/78GRfzMGRgYGBgYGBgYGQAAMkQNOQAAHwAAAAABJRU5ErkJggg=="

        response_data = {
            "data": [
                {"b64_json": b64_image_1},
                {"b64_json": b64_image_2},
            ]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)

        model_response = ImageResponse()

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == b64_image_1
        assert result.data[1].b64_json == b64_image_2

    def test_transform_image_generation_response_empty_data(self):
        """Test image generation response with empty data array."""
        response_data = {"data": []}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)

        model_response = ImageResponse()

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 0

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
        assert headers["Content-Type"] == "application/json"

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
        """Test that supported params include standard image generation parameters."""
        supported = self.config.get_supported_openai_params(model=self.model)
        assert "n" in supported
        assert "size" in supported
        assert "response_format" in supported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
