"""
Unit Tests for hosted_vllm Batches and Files API

Tests the integration of hosted_vllm provider with LiteLLM's batch and file operations.
Tests against a real OpenAI-compatible endpoint.
"""
import json
import os
import sys
import time
import uuid

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm


SERVER_URL = "https://exampleopenaiendpoint-production-0ee2.up.railway.app/v1"


@pytest.mark.asyncio()
@pytest.mark.skip(reason="Local only test")
async def test_hosted_vllm_full_workflow():
    """
    Test the complete workflow: create file -> create batch -> retrieve batch -> retrieve file.
    Tests against real OpenAI-compatible endpoint.
    """
    litellm._turn_on_debug()
    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    
    # Step 1: Create file
    print("\n=== Step 1: Creating file ===")
    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider="hosted_vllm",
        api_base=SERVER_URL,
        api_key="test-api-key",
    )
    
    print(f"✓ Created file: {file_obj.id}")
    assert file_obj.id is not None
    assert file_obj.object == "file"
    assert file_obj.purpose == "batch"
    
    # Step 2: Create batch
    print("\n=== Step 2: Creating batch ===")
    batch_obj = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="hosted_vllm",
        metadata={"test": "hosted_vllm_integration"},
        api_base=SERVER_URL,
        api_key="test-api-key",
    )
    
    print(f"✓ Created batch: {batch_obj.id}")
    print(f"  Status: {batch_obj.status}")
    print(f"  Input file: {batch_obj.input_file_id}")
    assert batch_obj.id is not None
    assert batch_obj.object == "batch"
    assert batch_obj.input_file_id == file_obj.id
    assert batch_obj.endpoint == "/v1/chat/completions"
    
    # Step 3: Retrieve batch
    print("\n=== Step 3: Retrieving batch ===")
    retrieved_batch = await litellm.aretrieve_batch(
        batch_id=batch_obj.id,
        custom_llm_provider="hosted_vllm",
        api_base=SERVER_URL,
        api_key="test-api-key",
    )
    
    print(f"✓ Retrieved batch: {retrieved_batch.id}")
    print(f"  Status: {retrieved_batch.status}")
    print(f"  Output file: {retrieved_batch.output_file_id}")
    assert retrieved_batch.id == batch_obj.id
    assert retrieved_batch.object == "batch"
    assert retrieved_batch.input_file_id == file_obj.id
    
    # Step 4: Retrieve file (verify file still accessible)
    print("\n=== Step 4: Retrieving original file ===")
    retrieved_file = await litellm.afile_retrieve(
        file_id=file_obj.id,
        custom_llm_provider="hosted_vllm",
        api_base=SERVER_URL,
        api_key="test-api-key",
    )
    
    print(f"✓ Retrieved file: {retrieved_file.id}")
    print(f"  Filename: {retrieved_file.filename}")
    print(f"  Bytes: {retrieved_file.bytes}")
    assert retrieved_file.id == file_obj.id
    assert retrieved_file.object == "file"
    
    print("\n✅ Full workflow test completed successfully!")
