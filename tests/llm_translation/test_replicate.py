"""
Unit tests for Replicate provider, particularly testing DeepSeek models
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion
from litellm.llms.replicate.chat.handler import (
    async_completion,
    completion as replicate_completion,
)


class TestReplicateStartingStatus:
    """Test that Replicate handler correctly handles 'starting' status for DeepSeek models"""

    @pytest.mark.asyncio
    @patch("litellm.llms.replicate.chat.handler.get_async_httpx_client")
    async def test_async_completion_handles_starting_status(
        self, mock_get_client
    ):
        """Test that async completion polls correctly when status is 'starting'"""
        # Mock the async HTTP client
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Mock the initial POST response (creates prediction)
        post_response = Mock()
        post_response.json.return_value = {
            "id": "test-prediction-id",
            "urls": {
                "get": "https://api.replicate.com/v1/predictions/test-id",
                "cancel": "https://api.replicate.com/v1/predictions/test-id/cancel",
            },
        }
        mock_client.post = AsyncMock(return_value=post_response)

        # Mock GET responses - first 'starting', then 'processing', then 'succeeded'
        get_response_starting = Mock()
        get_response_starting.status_code = 200
        get_response_starting.json.return_value = {
            "id": "test-prediction-id",
            "status": "starting",
            "output": None,
        }

        get_response_processing = Mock()
        get_response_processing.status_code = 200
        get_response_processing.json.return_value = {
            "id": "test-prediction-id",
            "status": "processing",
            "output": None,
        }

        get_response_succeeded = Mock()
        get_response_succeeded.status_code = 200
        get_response_succeeded.json.return_value = {
            "id": "test-prediction-id",
            "status": "succeeded",
            "output": ["Hello", " from", " DeepSeek!"],
        }
        get_response_succeeded.text = json.dumps(
            get_response_succeeded.json.return_value
        )
        get_response_succeeded.headers = {}

        # Configure mock to return different responses on successive calls
        mock_client.get = AsyncMock(
            side_effect=[
                get_response_starting,
                get_response_processing,
                get_response_succeeded,
            ]
        )

        # Create mock model response
        model_response = litellm.ModelResponse()
        model_response.choices = [litellm.Choices()]
        model_response.choices[0].message = litellm.Message(content="")

        # Create mock logging object
        mock_logging = Mock()
        mock_logging.post_call = Mock()

        # Call async_completion
        result = await async_completion(
            model_response=model_response,
            model="deepseek-ai/deepseek-v3",
            messages=[{"role": "user", "content": "Hi"}],
            encoding=None,
            optional_params={},
            litellm_params={},
            version_id="deepseek-ai/deepseek-v3",
            input_data={"input": {"prompt": "test"}},
            api_key="test-key",
            api_base="https://api.replicate.com",
            logging_obj=mock_logging,
            print_verbose=print,
            headers={"Authorization": "Token test-key"},
        )

        # Assert that we got responses
        assert result is not None
        assert result.choices[0].message.content == "Hello from DeepSeek!"
        
        # Verify that GET was called 3 times (starting, processing, succeeded)
        assert mock_client.get.call_count == 3

    @patch("litellm.llms.replicate.chat.handler._get_httpx_client")
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

    def test_transform_response_list_output(self):
        """Test standard list output format"""
        from litellm.llms.replicate.chat.transformation import ReplicateConfig

        config = ReplicateConfig()
        
        # Mock response with list output
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "succeeded",
            "output": ["Hello", " ", "world"],
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.headers = {}

        model_response = litellm.ModelResponse()
        model_response.choices = [litellm.Choices()]
        model_response.choices[0].message = litellm.Message(content="")

        mock_logging = Mock()
        mock_logging.post_call = Mock()

        result = config.transform_response(
            model="meta/llama-2-70b-chat",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={"input": {"prompt": "test"}},
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key",
        )

        assert result.choices[0].message.content == "Hello world"

    def test_transform_response_string_output(self):
        """Test string output format (as used by some DeepSeek models)"""
        from litellm.llms.replicate.chat.transformation import ReplicateConfig

        config = ReplicateConfig()
        
        # Mock response with string output
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "succeeded",
            "output": "Hello from DeepSeek",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_response.headers = {}

        model_response = litellm.ModelResponse()
        model_response.choices = [litellm.Choices()]
        model_response.choices[0].message = litellm.Message(content="")

        mock_logging = Mock()
        mock_logging.post_call = Mock()

        result = config.transform_response(
            model="deepseek-ai/deepseek-v3",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={"input": {"prompt": "test"}},
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key="test-key",
        )

        assert result.choices[0].message.content == "Hello from DeepSeek"


# Integration test (requires actual API key - skip in CI)
@pytest.mark.skip(reason="Requires REPLICATE_API_KEY environment variable")
def test_replicate_deepseek_integration():
    """Integration test with actual DeepSeek model on Replicate"""
    try:
        response = completion(
            model="replicate/deepseek-ai/deepseek-v3",
            messages=[{"role": "user", "content": "Say 'Hello World' and nothing else"}],
            max_tokens=20,
        )
        
        assert response is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
