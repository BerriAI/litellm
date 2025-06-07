import json
import os
import sys
import traceback
from unittest import mock
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import Mock

from litellm.proxy.pass_through_endpoints.common_utils import get_litellm_virtual_key, encode_bedrock_runtime_modelid_arn


@pytest.mark.asyncio
async def test_get_litellm_virtual_key():
    """
    Test that the get_litellm_virtual_key function correctly handles the API key authentication
    """
    # Test with x-litellm-api-key
    mock_request = Mock()
    mock_request.headers = {"x-litellm-api-key": "test-key-123"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"

    # Test with Authorization header
    mock_request.headers = {"Authorization": "Bearer auth-key-456"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer auth-key-456"

    # Test with both headers (x-litellm-api-key should take precedence)
    mock_request.headers = {
        "x-litellm-api-key": "test-key-123",
        "Authorization": "Bearer auth-key-456",
    }
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"



def test_encode_bedrock_runtime_modelid_arn():
    """
    Test that the encode_bedrock_modelid_arn function correctly encodes slashes in model IDs
    
    modelID names are generated based on the modelID regex as specified in AWS Bedrock Runtime API docs
    """
    # Test cases for each resource type
    test_cases = [
        # Application inference profile
        {
            "input": "/model/arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/my-app-profile-123/invoke",
            "expected": "/model/arn:aws:bedrock:us-west-2:123456789012:application-inference-profile%2Fmy-app-profile-123/invoke"
        },
        # Inference profile
        {
            "input": "/model/arn:aws:bedrock:ap-southeast-1:123456789012:inference-profile/apac.anthropic.claude-3-5-sonnet/invoke",
            "expected": "/model/arn:aws:bedrock:ap-southeast-1:123456789012:inference-profile%2Fapac.anthropic.claude-3-5-sonnet/invoke"
        },
        # Custom model (has 2 slashes)
        {
            "input": "/model/arn:aws:bedrock:us-east-1:123456789012:custom-model/anthropic.claude-3-sonnet.v1/abc123def456/converse",
            "expected": "/model/arn:aws:bedrock:us-east-1:123456789012:custom-model%2Fanthropic.claude-3-sonnet.v1%2Fabc123def456/converse"
        },
        # Foundation model
        {
            "input": "/model/arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-pro-v1.0/invoke",
            "expected": "/model/arn:aws:bedrock:us-west-2::foundation-model%2Famazon.nova-pro-v1.0/invoke"
        },
        # Imported model
        {
            "input": "/model/arn:aws:bedrock:eu-west-1:123456789012:imported-model/xyz789abc123/converse-stream",
            "expected": "/model/arn:aws:bedrock:eu-west-1:123456789012:imported-model%2Fxyz789abc123/converse-stream"
        },
        # Provisioned model
        {
            "input": "/model/arn:aws:bedrock:us-east-1:123456789012:provisioned-model/def456ghi789/invoke",
            "expected": "/model/arn:aws:bedrock:us-east-1:123456789012:provisioned-model%2Fdef456ghi789/invoke"
        },
        # Prompt
        {
            "input": "/model/arn:aws:bedrock:us-west-2:123456789012:prompt/abc123def4:1/invoke",
            "expected": "/model/arn:aws:bedrock:us-west-2:123456789012:prompt%2Fabc123def4:1/invoke"
        },
        # Endpoint (SageMaker)
        {
            "input": "/model/arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-custom-endpoint/invoke",
            "expected": "/model/arn:aws:sagemaker:us-east-1:123456789012:endpoint%2Fmy-custom-endpoint/invoke"
        },
        # Prompt router
        {
            "input": "/model/arn:aws:bedrock:us-west-2:123456789012:prompt-router/my-router-123/invoke",
            "expected": "/model/arn:aws:bedrock:us-west-2:123456789012:prompt-router%2Fmy-router-123/invoke"
        },
        # Default prompt router
        {
            "input": "/model/arn:aws:bedrock:eu-central-1:123456789012:default-prompt-router/default-router-456/converse",
            "expected": "/model/arn:aws:bedrock:eu-central-1:123456789012:default-prompt-router%2Fdefault-router-456/converse"
        },
    ]

    # base cases, should not have encoding of any sorts
    base_cases = [
        {
            "input": "/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            "expected": "/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke"
        },
        {
            "input": "/model/amazon.nova-pro-v1:0/converse",
            "expected": "/model/amazon.nova-pro-v1:0/converse"
        }
    ]
    
    # Run all test cases
    all_test_cases = test_cases + base_cases
    for test_case in all_test_cases:
        result = encode_bedrock_runtime_modelid_arn(test_case["input"])
        assert result == test_case["expected"], f"Failed for input: {test_case['input']}"
