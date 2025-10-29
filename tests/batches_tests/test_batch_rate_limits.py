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
from litellm.proxy.hooks.batch_rate_limiter import _PROXY_BatchRateLimiter, BatchFileUsage


def get_expected_batch_file_usage(file_path: str) -> tuple[int, int]:
    """
    Helper function to calculate expected request count and token count from a batch JSONL file.
    
    Returns:
        tuple[int, int]: (expected_request_count, expected_total_tokens)
    """
    with open(file_path, 'r') as f:
        file_contents = [json.loads(line) for line in f if line.strip()]
    
    expected_request_count = len(file_contents)
    expected_total_tokens = 0
    
    for item in file_contents:
        body = item.get("body", {})
        model = body.get("model", "")
        messages = body.get("messages", [])
        if messages:
            item_tokens = litellm.token_counter(model=model, messages=messages)
            expected_total_tokens += item_tokens
    
    return expected_request_count, expected_total_tokens


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
    

    assert file_obj.id is not None, "File ID should not be None"
    
    # Give API a moment to process the file
    await asyncio.sleep(1)
    
    
    # Count requests and token usage in input file
    tracked_batch_file_usage: BatchFileUsage = await BATCH_LIMITER.count_input_file_usage(
        file_id=file_obj.id,
        custom_llm_provider=CUSTOM_LLM_PROVIDER,
    )
    print(f"Actual total tokens: {tracked_batch_file_usage.total_tokens}")
    print(f"Actual request count: {tracked_batch_file_usage.request_count}")

    # Calculate expected values by reading the JSONL file
    expected_request_count, expected_total_tokens = get_expected_batch_file_usage(file_path=file_path)
    
    print(f"Expected request count: {expected_request_count}")
    print(f"Expected total tokens: {expected_total_tokens}")
    
    # Verify token counting results
    assert tracked_batch_file_usage.request_count == expected_request_count, f"Expected {expected_request_count} requests, got {tracked_batch_file_usage.request_count}"
    assert tracked_batch_file_usage.total_tokens == expected_total_tokens, f"Expected {expected_total_tokens} total_tokens, got {tracked_batch_file_usage.total_tokens}"
