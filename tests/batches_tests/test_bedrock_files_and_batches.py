
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
        aws_batch_role_arn="arn:aws:iam::888602223428:role/service-role/AmazonBedrockExecutionRoleForAgents_BB9HNW6V4CV"
    )
    print("CREATED BATCH RESPONSE=", create_batch_response)

