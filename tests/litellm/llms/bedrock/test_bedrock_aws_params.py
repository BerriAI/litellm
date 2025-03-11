import sys
import os
import io
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import AmazonAnthropicClaude3Config

@pytest.mark.parametrize(
    "model",
    [
        "bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
    ],
)
def test_bedrock_aws_params_not_in_request(model):
    """
    Test to ensure AWS parameters are not included in the request body sent to Bedrock
    """
    client = HTTPHandler()

    with patch.object(client, "post") as mock_client_post:
        # Mock response for Anthropic Claude on Bedrock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "id": "test-id",
            "content": [{"type": "text", "text": "This is a test response"}],
            "model": model,
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        mock_client_post.return_value = mock_response

        # Call completion with AWS parameters
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "What's AWS?"}],
            client=client,
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-west-2",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-west-2.amazonaws.com",
        )

        # Check that the request was made
        mock_client_post.assert_called_once()

        # Get the request body
        request_body = json.loads(mock_client_post.call_args.kwargs["data"])

        # Check that AWS parameters are not in the request body
        aws_params = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "aws_bedrock_runtime_endpoint",
        ]

        for param in aws_params:
            assert param not in request_body, f"{param} should not be in request body"

def test_transform_request_with_aws_params():
    """
    Test to verify that AmazonAnthropicClaude3Config.transform_request filters out AWS parameters
    """
    # Create an instance of AmazonAnthropicClaude3Config
    config = AmazonAnthropicClaude3Config()

    # Create test parameters with AWS parameters
    optional_params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_region_name": "us-west-2",
        "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
    }

    # Create a copy of the original params for comparison
    original_params = copy.deepcopy(optional_params)

    # Mock AnthropicConfig.transform_request to return the optional_params it receives
    with patch('litellm.llms.anthropic.chat.transformation.AnthropicConfig.transform_request') as mock_transform:
        mock_transform.return_value = {"mock": "response"}

        # Call the transform_request method
        config.transform_request(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )

        # Get the optional_params passed to AnthropicConfig.transform_request
        passed_params = mock_transform.call_args[1]["optional_params"]

        # Check that AWS parameters are not in the passed params
        aws_params = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "aws_bedrock_runtime_endpoint",
        ]

        for param in aws_params:
            assert param not in passed_params, f"{param} should not be in passed params"

        # Check that non-AWS parameters are still in the passed params
        assert "temperature" in passed_params
        assert "max_tokens" in passed_params

        # Verify that the original optional_params still contains AWS parameters (not modified)
        for param in aws_params:
            if param in original_params:
                assert param in optional_params, f"{param} should still be in original optional_params"


def test_transform_request_without_fix():
    """
    Test to demonstrate the issue without the fix - this shows what would happen
    if we didn't filter out AWS parameters
    """
    # Create an instance of AmazonAnthropicClaude3Config
    config = AmazonAnthropicClaude3Config()

    # Create test parameters with AWS parameters
    optional_params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_region_name": "us-west-2",
        "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
    }

    # Mock AnthropicConfig.transform_request to return the optional_params it receives
    with patch('litellm.llms.anthropic.chat.transformation.AnthropicConfig.transform_request') as mock_transform:
        mock_transform.return_value = {"mock": "response"}

        # Save the original method
        original_method = config.transform_request

        # Create a simple function that just passes through the parameters without filtering
        def mock_transform_without_filtering(*args, **kwargs):
            # Extract optional_params from kwargs
            optional_params = kwargs.get("optional_params", {})
            # Call the original AnthropicConfig.transform_request directly
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig
            return AnthropicConfig.transform_request(*args, **kwargs)

        try:
            # Replace the method with our mock
            config.transform_request = mock_transform_without_filtering

            # Call the method (which now doesn't filter AWS params)
            config.transform_request(
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params=optional_params,
                litellm_params={},
                headers={}
            )

            # Get the optional_params passed to AnthropicConfig.transform_request
            passed_params = mock_transform.call_args[1]["optional_params"]

            # Check that AWS parameters ARE in the passed params (demonstrating the issue)
            aws_params = [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_region_name",
                "aws_bedrock_runtime_endpoint",
            ]

            for param in aws_params:
                if param in optional_params:
                    assert param in passed_params, f"{param} should be in passed params (demonstrating the issue)"

        finally:
            # Restore the original method
            config.transform_request = original_method


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
    ],
)
async def test_bedrock_aws_params_not_in_request_async(model):
    """
    Test to ensure AWS parameters are not included in the request body sent to Bedrock (async version)
    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        # Mock response for Anthropic Claude on Bedrock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "id": "test-id",
            "content": [{"type": "text", "text": "This is a test response"}],
            "model": model,
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        mock_client_post.return_value = mock_response

        # Call completion with AWS parameters
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "What's AWS?"}],
            client=client,
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-west-2",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-west-2.amazonaws.com",
        )

        # Check that the request was made
        mock_client_post.assert_called_once()

        # Get the request body
        request_body = json.loads(mock_client_post.call_args.kwargs["data"])

        # Check that AWS parameters are not in the request body
        aws_params = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "aws_bedrock_runtime_endpoint",
        ]

        for param in aws_params:
            assert param not in request_body, f"{param} should not be in request body"