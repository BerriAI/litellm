
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

