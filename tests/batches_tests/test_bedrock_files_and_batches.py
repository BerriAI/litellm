
# What is this?
## Unit Tests for OpenAI Batches API
import asyncio
import json
import os
import sys
import traceback
import tempfile
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import pytest
from typing import Optional
import litellm
from unittest.mock import patch, MagicMock
import httpx


@pytest.mark.asyncio()
async def test_async_create_file():
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    litellm._turn_on_debug()
    print("Testing async create batch")

    file_name = "bedrock_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider="bedrock",
        s3_bucket_name="litellm-proxy",
    )

@pytest.mark.asyncio()
async def test_async_file_and_batch():
    """
    Test file retrieval
    """
    litellm._turn_on_debug()
    file_name = "bedrock_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider="bedrock",
        s3_bucket_name="litellm-proxy",
    )
    print("CREATED FILE RESPONSE=", file_obj)

    # create batch
    create_batch_response = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        metadata={"key1": "value1", "key2": "value2"},
        custom_llm_provider="bedrock",
        
        #########################################################
        # bedrock specific params
        #########################################################
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_batch_role_arn="arn:aws:iam::888602223428:role/service-role/AmazonBedrockExecutionRoleForAgents_BB9HNW6V4CV"
    )
    print("CREATED BATCH RESPONSE=", create_batch_response)

    # retrieve batch
    retrieve_batch_response = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id,
        custom_llm_provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    print("RETRIEVED BATCH RESPONSE=", retrieve_batch_response)
    
    # Validate the response
    assert retrieve_batch_response.id == create_batch_response.id
    assert retrieve_batch_response.object == "batch"
    assert retrieve_batch_response.status in ["validating", "in_progress", "completed", "failed", "cancelled"]


@pytest.mark.asyncio()
async def test_mock_bedrock_file_url_mapping():
    """
    Simple test to capture PUT URL and validate mapping to file ID.
    """
    print("Testing Bedrock file URL mapping")
    
    captured_put_url = None
    
    async def mock_async_create_file(transformed_request, **kwargs):
        nonlocal captured_put_url
        # Capture PUT URL from transformed request
        if isinstance(transformed_request, dict) and "url" in transformed_request:
            captured_put_url = transformed_request["url"]
        
        # Call the real method to get actual response
        from litellm.files.main import base_llm_http_handler
        return await base_llm_http_handler.__class__.async_create_file(
            base_llm_http_handler, transformed_request, **kwargs
        )
    
    with patch('litellm.files.main.base_llm_http_handler.async_create_file', side_effect=mock_async_create_file):
        file_obj = await litellm.acreate_file(
            file=open(os.path.join(os.path.dirname(__file__), "bedrock_batch_completions.jsonl"), "rb"),
            purpose="batch",
            custom_llm_provider="bedrock",
            s3_bucket_name="litellm-proxy",
        )
        
        print(f"PUT URL: {captured_put_url}")
        print(f"File ID: {file_obj.id}")
        
        # Validate URL was captured and response is correct
        assert captured_put_url is not None
        assert file_obj.id.startswith("s3://")
        
        # Verify mapping
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        bedrock_config = BedrockFilesConfig()
        expected_s3_uri, _ = bedrock_config._convert_https_url_to_s3_uri(captured_put_url)
        assert file_obj.id == expected_s3_uri


@pytest.mark.asyncio()
async def test_bedrock_retrieve_batch():
    """
    Test bedrock batch retrieval functionality, validating that input and output file IDs 
    are correctly extracted from the Bedrock response and included in the final transformed response.
    """
    print("Testing bedrock batch retrieval")
    
    # Mock bedrock batch response
    mock_bedrock_response = {
        "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123",
        "jobName": "test-job-123",
        "modelId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "roleArn": "arn:aws:iam::123456789012:role/service-role/AmazonBedrockExecutionRoleForAgents_TEST",
        "status": "InProgress",
        "message": "Job is in progress",
        "submitTime": "2024-01-01T12:00:00Z",
        "lastModifiedTime": "2024-01-01T12:30:00Z",
        "inputDataConfig": {
            "s3InputDataConfig": {
                "s3Uri": "s3://test-bucket/input/test-input.jsonl"
            }
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {
                "s3Uri": "s3://test-bucket/output/"
            }
        }
    }
    
    # Mock the HTTP response
    mock_response = MagicMock()
    mock_response.json.return_value = mock_bedrock_response
    mock_response.status_code = 200
    
    # Print the mock response to debug
    print("MOCK RESPONSE DATA:", mock_bedrock_response)
    
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get") as mock_get:
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test retrieve batch
        batch_response = await litellm.aretrieve_batch(
            batch_id="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        )
        
        print("MOCKED BATCH RESPONSE=", batch_response)
        
        # Validate the response
        assert batch_response.id == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123"
        assert batch_response.object == "batch"
        assert batch_response.status == "in_progress"  # Bedrock "InProgress" maps to "in_progress"
        assert batch_response.endpoint == "/v1/chat/completions"
        
        # Validate input and output file IDs in the final transformed response
        assert batch_response.input_file_id == "s3://test-bucket/input/test-input.jsonl"
        assert batch_response.output_file_id == "s3://test-bucket/output/"

