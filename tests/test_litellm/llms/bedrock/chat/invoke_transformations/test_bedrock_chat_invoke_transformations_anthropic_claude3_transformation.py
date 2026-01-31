import asyncio
import json
import os
import sys

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)


def test_get_supported_params_thinking():
    config = AmazonAnthropicClaudeConfig()
    params = config.get_supported_openai_params(
        model="anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert "thinking" in params


def test_aws_params_filtered_from_request_body():
    """
    Test that AWS authentication parameters are filtered out from the request body.
    
    This is a security test to ensure AWS credentials are not leaked in the request
    body sent to Bedrock. AWS params should only be used for request signing.
    
    Regression test for: AWS params (aws_role_name, aws_session_name, etc.) 
    being included in the Bedrock InvokeModel request body.
    """
    config = AmazonAnthropicClaudeConfig()
    
    # Test messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Optional params with AWS authentication parameters that should be filtered out
    optional_params = {
        # Regular Anthropic params - these SHOULD be in the request
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        
        # AWS authentication params - these should NOT be in the request body
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws_session_token": "FwoGZXIvYXdzEBYaDH...",
        "aws_region_name": "us-west-2",
        "aws_role_name": "arn:aws:iam::123456789012:role/test-role",
        "aws_session_name": "test-session",
        "aws_profile_name": "default",
        "aws_web_identity_token": "token123",
        "aws_sts_endpoint": "https://sts.amazonaws.com",
        "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
        "aws_external_id": "external-id-123",
    }
    
    # Transform the request
    result = config.transform_request(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=messages,
        optional_params=optional_params.copy(),  # Copy to avoid mutation
        litellm_params={},
        headers={},
    )
    
    # Convert result to JSON string to check what would be sent in the request
    result_json = json.dumps(result)
    
    # Verify AWS authentication params are NOT in the request body
    assert "aws_access_key_id" not in result_json, "AWS access key should not be in request body"
    assert "aws_secret_access_key" not in result_json, "AWS secret key should not be in request body"
    assert "aws_session_token" not in result_json, "AWS session token should not be in request body"
    assert "aws_region_name" not in result_json, "AWS region should not be in request body"
    assert "aws_role_name" not in result_json, "AWS role name should not be in request body"
    assert "aws_session_name" not in result_json, "AWS session name should not be in request body"
    assert "aws_profile_name" not in result_json, "AWS profile name should not be in request body"
    assert "aws_web_identity_token" not in result_json, "AWS web identity token should not be in request body"
    assert "aws_sts_endpoint" not in result_json, "AWS STS endpoint should not be in request body"
    assert "aws_bedrock_runtime_endpoint" not in result_json, "AWS bedrock endpoint should not be in request body"
    assert "aws_external_id" not in result_json, "AWS external ID should not be in request body"
    
    # Also check that the sensitive values themselves are not in the response
    assert "AKIAIOSFODNN7EXAMPLE" not in result_json, "AWS access key value leaked in request body"
    assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in result_json, "AWS secret key value leaked in request body"
    assert "arn:aws:iam::123456789012:role/test-role" not in result_json, "AWS role ARN leaked in request body"
    assert "test-session" not in result_json, "AWS session name leaked in request body"
    
    # Verify normal params ARE still in the request body
    assert result["max_tokens"] == 100, "max_tokens should be in request body"
    assert result["temperature"] == 0.7, "temperature should be in request body"
    assert result["top_p"] == 0.9, "top_p should be in request body"
    
    # Verify Bedrock-specific params are added
    assert result["anthropic_version"] == "bedrock-2023-05-31", "anthropic_version should be set"
    assert "model" not in result, "model should be removed for Bedrock Invoke API"
    assert "stream" not in result, "stream should be removed for Bedrock Invoke API"
    
    # Verify messages are present
    assert "messages" in result, "messages should be in request body"
    assert len(result["messages"]) == 1, "should have 1 message"
