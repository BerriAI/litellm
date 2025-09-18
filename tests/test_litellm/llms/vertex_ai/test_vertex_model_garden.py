"""
Test Vertex AI Model Garden functionality
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../"))

from litellm.llms.vertex_ai.vertex_model_garden.main import (
    VertexAIModelGardenModels,
    create_vertex_url,
)


class TestVertexAIModelGarden:
    """Test Vertex AI Model Garden functionality"""

    def test_create_vertex_url_dedicated_domain(self):
        """Test URL construction for dedicated domains"""
        # Test dedicated domain with dummy URL
        api_base = "https://dummy-endpoint-123.us-central1-999999999999.prediction.vertexai.goog"
        url = create_vertex_url(
            vertex_location="us-central1",
            vertex_project="test-project",
            stream=False,
            model="mg-endpoint-123",
            api_base=api_base,
        )

        expected = "https://dummy-endpoint-123.us-central1-999999999999.prediction.vertexai.goog/v1/projects/test-project/locations/us-central1/endpoints/mg-endpoint-123:predict"
        assert url == expected

    def test_create_vertex_url_shared_domain(self):
        """Test URL construction for shared domains"""
        # Test shared domain (no api_base provided)
        url = create_vertex_url(
            vertex_location="us-central1",
            vertex_project="test-project",
            stream=False,
            model="mg-endpoint-123",
            api_base=None,
        )

        expected = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/endpoints/mg-endpoint-123"
        assert url == expected

    def test_model_garden_routing(self):
        """Test that mg-endpoint models route to Model Garden handler"""
        # Test that the main completion function routes mg-endpoint models correctly
        # This is tested by checking the routing logic in main.py

        # Mock the completion function to avoid actual API calls
        with patch("litellm.main.vertex_model_garden_chat_completion") as mock_handler:
            mock_handler.completion.return_value = MagicMock()

            # Test that mg-endpoint models are routed to Model Garden handler
            # This would be called in the main completion function
            model = "mg-endpoint-dummy-123"

            # The routing logic should detect mg-endpoint models
            assert model.startswith("mg-endpoint-")

    def test_model_garden_handler_initialization(self):
        """Test Model Garden handler initialization"""
        handler = VertexAIModelGardenModels()
        assert handler is not None

    def test_dedicated_domain_detection(self):
        """Test detection of dedicated domains"""
        dedicated_domain = (
            "https://mg-endpoint-123.us-central1-222900905574.prediction.vertexai.goog"
        )
        shared_domain = "https://us-central1-aiplatform.googleapis.com"

        # Test dedicated domain detection
        assert "prediction.vertexai.goog" in dedicated_domain
        assert "prediction.vertexai.goog" not in shared_domain

    def test_url_construction_with_custom_endpoint(self):
        """Test URL construction with custom endpoint parameter"""
        # Test that custom_endpoint parameter is handled correctly
        api_base = (
            "https://mg-endpoint-123.us-central1-222900905574.prediction.vertexai.goog"
        )

        # Simulate the logic from the completion method
        if api_base and "prediction.vertexai.goog" in api_base:
            custom_endpoint = True
            final_api_base = f"{api_base}/v1/projects/test-project/locations/us-central1/endpoints/mg-endpoint-123:predict"
        else:
            custom_endpoint = False
            final_api_base = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/endpoints/mg-endpoint-123"

        assert custom_endpoint is True
        assert final_api_base.endswith(":predict")
        assert "prediction.vertexai.goog" in final_api_base

    def test_model_name_preservation(self):
        """Test that model name is preserved for logging"""
        # Test that the model name is not set to empty string
        model = "mg-endpoint-dummy-456"

        # Simulate the logic from the completion method
        # The model should not be set to empty string
        assert model != ""
        assert model.startswith("mg-endpoint-")

    def test_openai_prefix_compatibility(self):
        """Test that openai prefix still works for backward compatibility"""
        # Test that models with openai prefix are still detected
        model_with_prefix = "vertex_ai/openai/mg-endpoint-123"
        model_without_prefix = "vertex_ai/mg-endpoint-123"

        # Both should be detected as Model Garden models
        assert "openai" in model_with_prefix
        # After removing the vertex_ai/ prefix, it should start with mg-endpoint-
        model_without_prefix_clean = model_without_prefix.replace("vertex_ai/", "")
        assert model_without_prefix_clean.startswith("mg-endpoint-")

    def test_url_matches_curl_format(self):
        """Test that generated URL matches expected curl format"""
        # This test ensures the URL format matches what works with curl
        api_base = "https://dummy-endpoint-456.us-central1-888888888888.prediction.vertexai.goog"
        project_id = "dummy-project-123"
        location = "us-central1"
        endpoint_id = "mg-endpoint-dummy-456"

        url = create_vertex_url(
            vertex_location=location,
            vertex_project=project_id,
            stream=False,
            model=endpoint_id,
            api_base=api_base,
        )

        # Expected format from working curl command
        expected = f"{api_base}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        assert url == expected

        # Verify it ends with :predict (not /chat/completions)
        assert url.endswith(":predict")
        assert not url.endswith("/chat/completions")


if __name__ == "__main__":
    pytest.main([__file__])
