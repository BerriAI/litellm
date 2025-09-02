
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
async def test_async_create_batch():
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
    print("Response from creating file=", file_obj)
    retrieved_file = await litellm.afile_retrieve(
        file_id=file_obj.id,
        custom_llm_provider="bedrock",
    )
    print("Retrieved file=", retrieved_file)
    assert retrieved_file.id == file_obj.id
    assert retrieved_file.purpose == "batch"
    assert retrieved_file.custom_llm_provider == "bedrock"
    assert retrieved_file.s3_bucket_name == "litellm-proxy"