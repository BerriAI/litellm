"""
Integration Tests for Batch Rate Limits
"""

import asyncio
import json
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_rate_limiter import (
    BatchFileUsage,
    _PROXY_BatchRateLimiter,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache


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


@pytest.mark.asyncio()
async def test_batch_rate_limit_single_file():
    """
    Test batch rate limiting with a single file.
    
    Key has TPM = 200
    - File with < 200 tokens: should go through
    - File with > 200 tokens: should hit rate limit
    """
    import tempfile
    
    CUSTOM_LLM_PROVIDER = "openai"
    
    # Setup: Create internal usage cache and rate limiter
    dual_cache = DualCache()
    internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache
    )
    
    # Setup: Get batch rate limiter
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    assert batch_limiter is not None, "Batch rate limiter should be available"
    
    # Setup: Create user API key with TPM = 200
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key-123",
        tpm_limit=200,
        rpm_limit=10,
    )
    
    # Test 1: File with < 200 tokens should go through
    print("\n=== Test 1: File under 200 tokens ===")
    
    # Create a small batch file with ~150 tokens
    small_batch_content = """{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]}}
{"custom_id": "request-3", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hey"}]}}"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(small_batch_content)
        small_file_path = f.name
    
    try:
        # Upload file to OpenAI
        file_obj_small = await litellm.acreate_file(
            file=open(small_file_path, "rb"),
            purpose="batch",
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        print(f"Created small file: {file_obj_small.id}")
        await asyncio.sleep(1)  # Give API time to process
        
        data_under_limit = {
            "model": "gpt-3.5-turbo",
            "input_file_id": file_obj_small.id,
            "custom_llm_provider": CUSTOM_LLM_PROVIDER,
        }
        
        # Should not raise an exception
        result = await batch_limiter.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=data_under_limit,
            call_type="acreate_batch",
        )
        print(f"✓ File with ~150 tokens passed (under limit of 200)")
        print(f"  Actual tokens: {result.get('_batch_token_count')}")
    except HTTPException as e:
        pytest.fail(f"Should not have hit rate limit with small file: {e.detail}")
    finally:
        os.unlink(small_file_path)
    
    # Test 2: File with > 200 tokens should hit rate limit
    print("\n=== Test 2: File over 200 tokens ===")
    
    # Reset cache for clean test
    dual_cache = DualCache()
    internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache
    )
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    
    # Create a larger batch file with ~10000+ tokens (100x larger to ensure it exceeds 200 token limit)
    base_message = "This is a longer message that will consume more tokens from the rate limit. " * 100
    
    # Build JSONL content with json.dumps to avoid f-string nesting issues
    import json as json_lib
    requests = []
    for i in range(1, 4):
        request_obj = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": base_message}]
            }
        }
        requests.append(json_lib.dumps(request_obj))
    
    large_batch_content = "\n".join(requests)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(large_batch_content)
        large_file_path = f.name
    
    try:
        # Upload file to OpenAI
        file_obj_large = await litellm.acreate_file(
            file=open(large_file_path, "rb"),
            purpose="batch",
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        print(f"Created large file: {file_obj_large.id}")
        await asyncio.sleep(1)  # Give API time to process
        
        data_over_limit = {
            "model": "gpt-3.5-turbo",
            "input_file_id": file_obj_large.id,
            "custom_llm_provider": CUSTOM_LLM_PROVIDER,
        }
        
        # Should raise HTTPException with 429 status
        with pytest.raises(HTTPException) as exc_info:
            await batch_limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=dual_cache,
                data=data_over_limit,
                call_type="acreate_batch",
            )
        
        assert exc_info.value.status_code == 429, "Should return 429 status code"
        assert "tokens" in exc_info.value.detail.lower(), "Error message should mention tokens"
        print(f"✓ File with 250+ tokens correctly rejected (over limit of 200)")
        print(f"  Error: {exc_info.value.detail}")
    finally:
        os.unlink(large_file_path)


@pytest.mark.asyncio()
async def test_batch_rate_limit_multiple_requests():
    """
    Test batch rate limiting with multiple requests.
    
    Key has TPM = 200
    - Request 1: file with ~100 tokens (should go through, 100/200 used)
    - Request 2: file with ~105 tokens (should hit limit, 100+105=205 > 200)
    """
    import tempfile
    
    CUSTOM_LLM_PROVIDER = "openai"
    
    # Setup: Create internal usage cache and rate limiter
    dual_cache = DualCache()
    internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache
    )
    
    # Setup: Get batch rate limiter
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    assert batch_limiter is not None, "Batch rate limiter should be available"
    
    # Setup: Create user API key with TPM = 200
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key-456",
        tpm_limit=200,
        rpm_limit=10,
    )
    
    # Request 1: File with ~100 tokens
    print("\n=== Request 1: File with ~100 tokens ===")
    
    # Create file with ~100 tokens
    import json as json_lib
    message_1 = "This message has some content to reach about 100 tokens total. " * 4
    requests_1 = []
    for i in range(1, 3):
        request_obj = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": message_1}]
            }
        }
        requests_1.append(json_lib.dumps(request_obj))
    
    batch_content_1 = "\n".join(requests_1)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(batch_content_1)
        file_path_1 = f.name
    
    try:
        # Upload file to OpenAI
        file_obj_1 = await litellm.acreate_file(
            file=open(file_path_1, "rb"),
            purpose="batch",
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        print(f"Created file 1: {file_obj_1.id}")
        await asyncio.sleep(1)  # Give API time to process
        
        data_request1 = {
            "model": "gpt-3.5-turbo",
            "input_file_id": file_obj_1.id,
            "custom_llm_provider": CUSTOM_LLM_PROVIDER,
        }
        
        # Should not raise an exception
        result1 = await batch_limiter.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=data_request1,
            call_type="acreate_batch",
        )
        tokens_used_1 = result1.get('_batch_token_count', 0)
        print(f"✓ Request 1 with {tokens_used_1} tokens passed ({tokens_used_1}/200 used)")
    except HTTPException as e:
        pytest.fail(f"Request 1 should not have hit rate limit: {e.detail}")
    finally:
        os.unlink(file_path_1)
    
    # Request 2: File with ~105+ tokens (total would exceed 200)
    print("\n=== Request 2: File with ~105 tokens (should hit limit) ===")
    
    # Create file with ~105+ tokens
    message_2 = "This is another message with more content to exceed the remaining limit. " * 11
    requests_2 = []
    for i in range(1, 3):
        request_obj = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": message_2}]
            }
        }
        requests_2.append(json_lib.dumps(request_obj))
    
    batch_content_2 = "\n".join(requests_2)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(batch_content_2)
        file_path_2 = f.name
    
    try:
        # Upload file to OpenAI
        file_obj_2 = await litellm.acreate_file(
            file=open(file_path_2, "rb"),
            purpose="batch",
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        print(f"Created file 2: {file_obj_2.id}")
        await asyncio.sleep(1)  # Give API time to process
        
        data_request2 = {
            "model": "gpt-3.5-turbo",
            "input_file_id": file_obj_2.id,
            "custom_llm_provider": CUSTOM_LLM_PROVIDER,
        }
        
        # Should raise HTTPException with 429 status
        with pytest.raises(HTTPException) as exc_info:
            await batch_limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=dual_cache,
                data=data_request2,
                call_type="acreate_batch",
            )
        
        assert exc_info.value.status_code == 429, "Should return 429 status code"
        assert "tokens" in exc_info.value.detail.lower(), "Error message should mention tokens"
        print(f"✓ Request 2 correctly rejected")
        print(f"  Error: {exc_info.value.detail}")
    finally:
        os.unlink(file_path_2)
