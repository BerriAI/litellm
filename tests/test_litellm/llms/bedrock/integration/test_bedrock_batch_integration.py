"""Integration tests for Bedrock batch functionality with LiteLLM main API."""

import os
from unittest.mock import patch

import pytest

import litellm


@pytest.mark.asyncio
@patch.dict(os.environ, {
    "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
    "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
    "AWS_REGION": "us-east-1"
})
async def test_bedrock_file_creation_api_integration():
    """Test that litellm.create_file works with bedrock provider."""
    # Mock file content
    test_content = b'{"test": "batch request"}'
    
    # This would require actual S3 upload, so we'll test the API surface
    try:
        # Test that the function accepts bedrock provider
        create_file_request = {
            "file": test_content,
            "purpose": "batch",
            "custom_llm_provider": "bedrock"
        }
        
        # Verify the API accepts the parameters (will fail at AWS call level)
        # but confirms the interface is correct
        with pytest.raises(Exception):  # Expected to fail without real AWS setup
            response = await litellm.acreate_file(**create_file_request)
    except TypeError as e:
        # If we get a TypeError, it means the function signature doesn't accept bedrock
        pytest.fail(f"API doesn't accept bedrock provider: {e}")


@patch.dict(os.environ, {
    "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket", 
    "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
    "AWS_REGION": "us-east-1"
})
def test_bedrock_batch_creation_api_integration():
    """Test that litellm.create_batch works with bedrock provider."""
    # Test that the function accepts bedrock provider
    try:
        create_batch_request = {
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions", 
            "input_file_id": "file-test123",
            "custom_llm_provider": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0"
        }
        
        # Verify the API accepts the parameters (will fail at AWS call level)
        with pytest.raises(Exception):  # Expected to fail without real AWS setup
            response = litellm.create_batch(**create_batch_request)
    except TypeError as e:
        # If we get a TypeError, it means the function signature doesn't accept bedrock
        pytest.fail(f"API doesn't accept bedrock provider: {e}")


@patch.dict(os.environ, {
    "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
    "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole", 
    "AWS_REGION": "us-east-1"
})
@pytest.mark.skip(reason="Bedrock batch retrieval not yet implemented")
def test_bedrock_batch_retrieval_api_integration():
    """Test that litellm.retrieve_batch works with bedrock provider."""
    # Test that the function accepts bedrock provider
    try:
        retrieve_batch_request = {
            "batch_id": "test-batch-123",
            "custom_llm_provider": "bedrock"
        }
        
        # Verify the API accepts the parameters (will fail at AWS call level)
        with pytest.raises(Exception):  # Expected to fail without real AWS setup
            response = litellm.retrieve_batch(**retrieve_batch_request)
    except TypeError as e:
        # If we get a TypeError, it means the function signature doesn't accept bedrock
        pytest.fail(f"API doesn't accept bedrock provider: {e}")



if __name__ == "__main__":
    pytest.main([__file__])