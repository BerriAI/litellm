"""
Unit tests for Replicate provider, particularly testing DeepSeek models
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm
from litellm.llms.replicate.chat.handler import (
    async_completion,
)
from litellm.llms.replicate.chat.handler import (
    completion as replicate_completion,
)


class TestReplicateStartingStatus:
    """Test that Replicate handler correctly handles 'starting' status for DeepSeek models"""


    @patch("litellm.llms.replicate.chat.handler.get_httpx_client")
    def test_sync_completion_handles_starting_status(self, mock_get_client):
        """Test that sync completion polls correctly when status is 'starting'"""
        # Mock the sync HTTP client
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock the initial POST response
        post_response = Mock()
        post_response.json.return_value = {
            "id": "test-prediction-id",
            "urls": {
                "get": "https://api.replicate.com/v1/predictions/test-id",
                "cancel": "https://api.replicate.com/v1/predictions/test-id/cancel",
            },
        }
        mock_client.post.return_value = post_response

        # Mock GET responses
        get_response_starting = Mock()
        get_response_starting.status_code = 200
        get_response_starting.json.return_value = {
            "id": "test-prediction-id",
            "status": "starting",
            "output": None,
        }

        get_response_succeeded = Mock()
        get_response_succeeded.status_code = 200
        get_response_succeeded.json.return_value = {
            "id": "test-prediction-id",
            "status": "succeeded",
            "output": ["Hello", " DeepSeek!"],
        }
        get_response_succeeded.text = json.dumps(
            get_response_succeeded.json.return_value
        )
        get_response_succeeded.headers = {}

        # Configure mock to return different responses
        mock_client.get.side_effect = [get_response_starting, get_response_succeeded]

        # Create mock objects
        model_response = litellm.ModelResponse()
        model_response.choices = [litellm.Choices()]
        model_response.choices[0].message = litellm.Message(content="")

        mock_logging = Mock()
        mock_logging.post_call = Mock()

        # Call completion with mock_response to avoid actual API call
        with patch("time.sleep"):  # Skip sleep delays in test
            result = replicate_completion(
                model="deepseek-ai/deepseek-v3",
                messages=[{"role": "user", "content": "Hi"}],
                api_base="https://api.replicate.com",
                model_response=model_response,
                print_verbose=print,
                optional_params={},
                litellm_params={},
                logging_obj=mock_logging,
                api_key="test-key",
                encoding=None,
                headers={},
            )

        # Assert results
        assert result is not None
        assert result.choices[0].message.content == "Hello DeepSeek!"

        # Verify GET was called multiple times
        assert mock_client.get.call_count >= 1


class TestReplicateOutputFormats:
    """Test that Replicate handler handles different output formats from models"""




# Integration test (requires actual API key - skip in CI)
