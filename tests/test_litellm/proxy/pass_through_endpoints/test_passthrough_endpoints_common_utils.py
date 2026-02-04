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

from litellm.passthrough.utils import CommonUtils, HttpPassThroughEndpointHelpers

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import Mock

from litellm.proxy.pass_through_endpoints.common_utils import get_litellm_virtual_key


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
    # Test application-inference-profile ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile/r742sbn2zckd/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789123:application-inference-profile%2Fr742sbn2zckd/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected
    
    # Test inference-profile ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:inference-profile/test-profile/invoke"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Ftest-profile/invoke"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected
    
    # Test foundation-model ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:foundation-model%2Fanthropic.claude-3/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected
    
    # Test custom-model ARN (2 slashes)
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model.fine-tuned/abc123/invoke"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:custom-model%2Fmy-model.fine-tuned%2Fabc123/invoke"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected
    
    # Test provisioned-model ARN
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:provisioned-model/test-model/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:provisioned-model%2Ftest-model/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected


def test_encode_bedrock_runtime_modelid_arn_no_arn():
    # Test regular model ID (no ARN)
    endpoint = "model/anthropic.claude-3-sonnet-20240229-v1:0/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == endpoint


def test_encode_bedrock_runtime_modelid_arn_edge_cases():
    # Test multiple ARN types (should only encode first match)
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test1/converse"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile%2Ftest1/converse"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected
    
    # Test ARN with special characters in resource ID
    endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile.v1/invoke"
    expected = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile%2Ftest-profile.v1/invoke"
    result = CommonUtils.encode_bedrock_runtime_modelid_arn(endpoint)
    assert result == expected


def test_forward_headers_strips_litellm_api_key():
    """x-litellm-api-key should not be forwarded to upstream providers."""
    request_headers = {
        "x-litellm-api-key": "sk-litellm-secret-key",
        "content-type": "application/json",
        "x-api-key": "sk-ant-api-key",
    }

    result = HttpPassThroughEndpointHelpers.forward_headers_from_request(
        request_headers=request_headers.copy(),
        headers={},
        forward_headers=True,
    )

    assert "x-litellm-api-key" not in result
    assert result.get("content-type") == "application/json"
    assert result.get("x-api-key") == "sk-ant-api-key"


def test_forward_headers_strips_host_and_content_length():
    """host and content-length should not be forwarded."""
    request_headers = {
        "host": "api.anthropic.com",
        "content-length": "1234",
        "content-type": "application/json",
    }

    result = HttpPassThroughEndpointHelpers.forward_headers_from_request(
        request_headers=request_headers.copy(),
        headers={},
        forward_headers=True,
    )

    assert "host" not in result
    assert "content-length" not in result
    assert result.get("content-type") == "application/json"