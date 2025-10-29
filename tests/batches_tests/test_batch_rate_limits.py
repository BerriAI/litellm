"""
Integration Tests for Batch Rate Limits
"""

import os
import sys
import traceback
import json
import pytest
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter

@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set - skipping integration test"
)
async def test_batch_rate_limits():
    """
    Integration test for batch rate limits with real OpenAI API calls.
    Tests the full flow: file creation -> token counting -> cleanup
    """
    litellm._turn_on_debug()
    CUSTOM_LLM_PROVIDER = "openai"
    BATCH_LIMITER = _PROXY_BatchRateLimiter(
        internal_usage_cache=None,
        parallel_request_limiter=None,
    )

    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)

    # Create file on OpenAI
    print(f"Creating file from {file_path}")
    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider=CUSTOM_LLM_PROVIDER,
    )
    print(f"Response from creating file: {file_obj}")
    
    try:
        assert file_obj.id is not None, "File ID should not be None"
        
        # Give API a moment to process the file
        await asyncio.sleep(1)
        
        # Count requests and token usage in input file
        total_tokens, request_count = await BATCH_LIMITER.count_input_file_usage(
            file_id=file_obj.id,
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        print(f"Total tokens: {total_tokens}")
        print(f"Request count: {request_count}")
        
        # Verify token counting results
        assert request_count == 2, f"Expected 2 requests, got {request_count}"
        assert total_tokens == 0, f"Expected 0 total_tokens (token counting not yet implemented), got {total_tokens}"
        
    finally:
        # Cleanup: delete the file
        try:
            print(f"Cleaning up file {file_obj.id}")
            await litellm.afile_delete(
                file_id=file_obj.id,
                custom_llm_provider=CUSTOM_LLM_PROVIDER,
            )
            print("File deleted successfully")
        except Exception as e:
            print(f"Warning: Failed to delete file {file_obj.id}: {e}")

    # batch_input_file_id = file_obj.id
    # assert (
    #     batch_input_file_id is not None
    # ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    # await asyncio.sleep(1)
    # create_batch_response = await litellm.acreate_batch(
    #     completion_window="24h",
    #     endpoint="/v1/chat/completions",
    #     input_file_id=batch_input_file_id,
    #     custom_llm_provider=provider,
    #     metadata={"key1": "value1", "key2": "value2"},
    # )