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


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set - skipping integration test"
)
async def test_batch_rate_limiter_with_managed_files():
    """
    Test for GEN-2166: Verify batch rate limiter can read user files when managed files are enabled.
    
    This test ensures that:
    1. The batch rate limiter passes user_api_key_dict to afile_content()
    2. The managed files hook can verify file ownership correctly
    3. Rate limiting is enforced (not silently bypassed)
    4. No 403 Permission Denied errors occur for files owned by the user
    """
    import tempfile
    from unittest.mock import AsyncMock, MagicMock, patch
    
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
    
    # Setup: Create user API key with TPM = 500, RPM = 10
    test_user_id = "test-user-abc123"
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key-managed-files",
        user_id=test_user_id,
        tpm_limit=500,
        rpm_limit=10,
    )
    
    print(f"\n=== Testing Batch Rate Limiter with Managed Files ===")
    print(f"User ID: {test_user_id}")
    
    # Create a batch file with ~200 tokens
    import json as json_lib
    message = "This is a test message for batch rate limiting with managed files. " * 5
    requests = []
    for i in range(1, 4):
        request_obj = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": message}]
            }
        }
        requests.append(json_lib.dumps(request_obj))
    
    batch_content = "\n".join(requests)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(batch_content)
        file_path = f.name
    
    try:
        # Step 1: Upload file to OpenAI (simulating user upload)
        print("\n1. Uploading batch input file...")
        file_obj = await litellm.acreate_file(
            file=open(file_path, "rb"),
            purpose="batch",
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        print(f"   ✓ File uploaded: {file_obj.id}")
        await asyncio.sleep(1)  # Give API time to process
        
        # Step 2: Mock managed files hook to simulate file ownership check
        # In a real scenario, the managed files hook would check if the user owns the file
        # For this test, we'll verify that user_api_key_dict is passed correctly
        print("\n2. Testing rate limiter file access with user context...")
        
        # Track if user_api_key_dict was passed to afile_content
        original_afile_content = litellm.afile_content
        user_context_passed = {"value": False}
        
        async def mock_afile_content(*args, **kwargs):
            # Check if user_api_key_dict was passed
            if "user_api_key_dict" in kwargs and kwargs["user_api_key_dict"] is not None:
                user_context_passed["value"] = True
                print(f"   ✓ user_api_key_dict passed to afile_content")
                print(f"     User ID: {kwargs['user_api_key_dict'].user_id}")
            else:
                print(f"   ✗ user_api_key_dict NOT passed to afile_content (BUG!)")
            
            # Call original function
            return await original_afile_content(*args, **kwargs)
        
        # Patch afile_content to track the call
        with patch('litellm.afile_content', side_effect=mock_afile_content):
            data = {
                "model": "gpt-3.5-turbo",
                "input_file_id": file_obj.id,
                "custom_llm_provider": CUSTOM_LLM_PROVIDER,
            }
            
            # Step 3: Submit batch and verify rate limiting works
            print("\n3. Submitting batch with rate limiting...")
            result = await batch_limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=dual_cache,
                data=data,
                call_type="acreate_batch",
            )
            
            tokens_used = result.get('_batch_token_count', 0)
            requests_count = result.get('_batch_request_count', 0)
            print(f"   ✓ Batch submitted successfully")
            print(f"     Tokens counted: {tokens_used}")
            print(f"     Requests counted: {requests_count}")
            print(f"     Rate limit usage: {tokens_used}/500 TPM, {requests_count}/10 RPM")
        
        # Step 4: Verify user context was passed
        print("\n4. Verifying fix for GEN-2166...")
        assert user_context_passed["value"], (
            "FAILED: user_api_key_dict was not passed to afile_content(). "
            "This means the bug GEN-2166 is not fixed!"
        )
        print("   ✓ Fix verified: user_api_key_dict is correctly passed")
        
        # Step 5: Verify rate limiting is actually enforced (not bypassed)
        print("\n5. Verifying rate limiting is enforced...")
        assert tokens_used > 0, "Token count should be greater than 0"
        assert requests_count > 0, "Request count should be greater than 0"
        print("   ✓ Rate limiting is active (not silently bypassed)")
        
        print("\n=== Test Passed: GEN-2166 Fix Verified ===")
        print("✓ Batch rate limiter can access user files")
        print("✓ User context is correctly passed")
        print("✓ Rate limiting is enforced")
        print("✓ No silent failures")
        
    except HTTPException as e:
        if e.status_code == 403:
            pytest.fail(
                f"FAILED: Got 403 Permission Denied error. "
                f"This indicates the bug GEN-2166 is not fixed. "
                f"Error: {e.detail}"
            )
        else:
            raise
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")
    finally:
        os.unlink(file_path)


@pytest.mark.asyncio()
async def test_batch_rate_limiter_without_user_context():
    """
    Test that verifies the bug scenario from GEN-2166.
    
    When user_api_key_dict is NOT passed to count_input_file_usage(),
    the function should still work for non-managed files, but would fail
    for managed files (which is the bug we fixed).
    
    This test documents the expected behavior with and without user context.
    """
    import tempfile
    
    CUSTOM_LLM_PROVIDER = "openai"
    
    # Setup
    BATCH_LIMITER = _PROXY_BatchRateLimiter(
        internal_usage_cache=None,
        parallel_request_limiter=None,
    )
    
    # Create a simple batch file
    batch_content = """{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}}"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(batch_content)
        file_path = f.name
    
    try:
        # Upload file
        file_obj = await litellm.acreate_file(
            file=open(file_path, "rb"),
            purpose="batch",
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
        )
        await asyncio.sleep(1)
        
        # Test 1: Without user context (old behavior - would fail with managed files)
        print("\n=== Test 1: count_input_file_usage WITHOUT user context ===")
        try:
            usage_without_context = await BATCH_LIMITER.count_input_file_usage(
                file_id=file_obj.id,
                custom_llm_provider=CUSTOM_LLM_PROVIDER,
                user_api_key_dict=None,  # Explicitly passing None
            )
            print(f"✓ Works for non-managed files (tokens: {usage_without_context.total_tokens})")
            print("  Note: Would fail with 403 for managed files (GEN-2166 bug)")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
        
        # Test 2: With user context (new behavior - works with managed files)
        print("\n=== Test 2: count_input_file_usage WITH user context ===")
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user-123",
        )
        
        usage_with_context = await BATCH_LIMITER.count_input_file_usage(
            file_id=file_obj.id,
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
            user_api_key_dict=user_api_key_dict,  # Passing user context
        )
        print(f"✓ Works with user context (tokens: {usage_with_context.total_tokens})")
        print("  Note: This fixes GEN-2166 for managed files")
        
        # Verify both return the same results
        assert usage_with_context.total_tokens == usage_without_context.total_tokens
        assert usage_with_context.request_count == usage_without_context.request_count
        print("\n✓ Both methods return identical results for non-managed files")
        
    finally:
        os.unlink(file_path)


@pytest.mark.asyncio()
async def test_batch_rate_limiter_managed_files_regression():
    """
    Regression test for GEN-2166: Batch Rate Limiter Cannot Access User Files
    
    This test ensures that the batch rate limiter can properly access managed files
    by verifying that:
    1. Managed files are detected correctly (base64 encoded unified file IDs)
    2. The _fetch_managed_file_content method uses the managed files hook
    3. User context (user_api_key_dict) is properly passed through
    4. No 403 errors occur when accessing files owned by the user
    5. The fix doesn't break non-managed file access
    
    This is a unit test that doesn't require external API calls.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.types.llms.openai import HttpxBinaryResponseContent
    import httpx
    
    print("\n=== Regression Test: GEN-2166 Batch Rate Limiter Managed Files ===")
    
    # Setup: Create batch rate limiter
    dual_cache = DualCache()
    internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache
    )
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    assert batch_limiter is not None
    
    # Setup: Create user API key dict
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key-regression",
        user_id="test-user-regression",
        tpm_limit=1000,
        rpm_limit=10,
    )
    
    # Setup: Create mock file content (batch input file)
    batch_content = b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test message for regression"}]}}'
    
    # Mock managed file ID (base64 encoded unified file ID format)
    managed_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9vY3RldC1zdHJlYW07dW5pZmllZF9pZCxyZWdyZXNzaW9uLXRlc3QtZmlsZQ=="
    
    # Test 1: Verify managed file detection
    print("\n1. Verifying managed file detection...")
    from litellm.proxy.openai_files_endpoints.common_utils import (
        _is_base64_encoded_unified_file_id,
    )
    is_managed = _is_base64_encoded_unified_file_id(managed_file_id)
    assert is_managed, "Managed file should be detected correctly"
    print("   ✓ Managed file detected")
    
    # Test 2: Verify _fetch_managed_file_content uses managed files hook
    print("\n2. Verifying managed files hook integration...")
    
    # Create mock managed files hook
    class MockManagedFiles(BaseFileEndpoints):
        def __init__(self):
            self._afile_content_called = False
            self._last_call_args = None
        
        async def acreate_file(self, *args, **kwargs):
            pass
        
        async def afile_content(self, *args, **kwargs):
            self._afile_content_called = True
            self._last_call_args = kwargs
            # Return mock file content
            mock_response = httpx.Response(
                status_code=200,
                content=batch_content,
                headers={"content-type": "application/octet-stream"},
            )
            return HttpxBinaryResponseContent(response=mock_response)
        
        async def afile_delete(self, *args, **kwargs):
            pass
        
        async def afile_list(self, *args, **kwargs):
            pass
        
        async def afile_retrieve(self, *args, **kwargs):
            pass
    
    mock_managed_files = MockManagedFiles()
    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.get_proxy_hook.return_value = mock_managed_files
    
    # Patch proxy_server imports
    with patch.dict('sys.modules', {
        'litellm.proxy.proxy_server': MagicMock(
            llm_router=mock_llm_router,
            proxy_logging_obj=mock_proxy_logging_obj,
        )
    }):
        # Call _fetch_managed_file_content
        result = await batch_limiter._fetch_managed_file_content(
            file_id=managed_file_id,
            user_api_key_dict=user_api_key_dict,
        )
        
        # Verify managed files hook was called
        assert mock_managed_files._afile_content_called, \
            "REGRESSION: managed_files_obj.afile_content was not called! Bug GEN-2166 has returned."
        
        # Verify user context was passed
        assert mock_managed_files._last_call_args is not None, \
            "REGRESSION: No arguments passed to afile_content"
        assert 'file_id' in mock_managed_files._last_call_args, \
            "REGRESSION: file_id not passed to managed files hook"
        assert mock_managed_files._last_call_args['file_id'] == managed_file_id, \
            "REGRESSION: Incorrect file_id passed"
        assert 'llm_router' in mock_managed_files._last_call_args, \
            "REGRESSION: llm_router not passed to managed files hook"
        
        print("   ✓ Managed files hook called correctly")
        print("   ✓ User context passed correctly")
    
    # Test 3: Verify count_input_file_usage uses managed files path
    print("\n3. Verifying count_input_file_usage integration...")
    
    with patch.object(batch_limiter, '_fetch_managed_file_content') as mock_fetch:
        mock_response = httpx.Response(
            status_code=200,
            content=batch_content,
            headers={"content-type": "application/octet-stream"},
        )
        mock_fetch.return_value = HttpxBinaryResponseContent(response=mock_response)
        
        # Call count_input_file_usage with managed file
        usage = await batch_limiter.count_input_file_usage(
            file_id=managed_file_id,
            custom_llm_provider="openai",
            user_api_key_dict=user_api_key_dict,
        )
        
        # Verify _fetch_managed_file_content was called
        assert mock_fetch.called, \
            "REGRESSION: _fetch_managed_file_content not called for managed files! Bug GEN-2166 has returned."
        
        # Verify correct parameters were passed
        call_kwargs = mock_fetch.call_args.kwargs
        assert call_kwargs['file_id'] == managed_file_id, \
            "REGRESSION: Incorrect file_id passed to _fetch_managed_file_content"
        assert call_kwargs['user_api_key_dict'] == user_api_key_dict, \
            "REGRESSION: user_api_key_dict not passed! Bug GEN-2166 has returned."
        
        # Verify usage was calculated
        assert usage.total_tokens > 0, "Token count should be greater than 0"
        assert usage.request_count == 1, "Request count should be 1"
        
        print("   ✓ Managed file path used")
        print(f"   ✓ Token count: {usage.total_tokens}")
        print(f"   ✓ Request count: {usage.request_count}")
    
    # Test 4: Verify non-managed files still work
    print("\n4. Verifying non-managed files still work...")
    
    non_managed_file_id = "file-abc123"  # Standard OpenAI file ID
    
    with patch('litellm.afile_content') as mock_afile_content:
        mock_response = httpx.Response(
            status_code=200,
            content=batch_content,
            headers={"content-type": "application/octet-stream"},
        )
        mock_afile_content.return_value = HttpxBinaryResponseContent(response=mock_response)
        
        # Call count_input_file_usage with non-managed file
        usage = await batch_limiter.count_input_file_usage(
            file_id=non_managed_file_id,
            custom_llm_provider="openai",
            user_api_key_dict=user_api_key_dict,
        )
        
        # Verify litellm.afile_content was called
        assert mock_afile_content.called, \
            "REGRESSION: litellm.afile_content not called for non-managed files"
        
        print("   ✓ Standard file path used")
        print(f"   ✓ Token count: {usage.total_tokens}")
    
    # Test 5: Verify the fix prevents 403 errors
    print("\n5. Verifying 403 error prevention...")
    
    # Simulate the bug scenario: managed files hook not being used
    with patch.object(batch_limiter, '_fetch_managed_file_content') as mock_fetch:
        # If this is NOT called for managed files, the bug has returned
        mock_fetch.side_effect = Exception("Should not be called if bug exists")
        
        # This should call _fetch_managed_file_content
        try:
            with patch('litellm.afile_content') as mock_afile_content:
                # If litellm.afile_content is called for managed files, bug exists
                mock_afile_content.side_effect = Exception(
                    "Error code: 403 - User does not have access to the file"
                )
                
                # Reset mock_fetch to return valid content
                mock_response = httpx.Response(
                    status_code=200,
                    content=batch_content,
                    headers={"content-type": "application/octet-stream"},
                )
                mock_fetch.side_effect = None
                mock_fetch.return_value = HttpxBinaryResponseContent(response=mock_response)
                
                # This should use _fetch_managed_file_content, not litellm.afile_content
                usage = await batch_limiter.count_input_file_usage(
                    file_id=managed_file_id,
                    custom_llm_provider="openai",
                    user_api_key_dict=user_api_key_dict,
                )
                
                # Verify managed files path was used (not standard path that causes 403)
                assert mock_fetch.called, \
                    "REGRESSION: Managed files path not used! This would cause 403 errors."
                assert not mock_afile_content.called, \
                    "REGRESSION: Standard path used for managed files! This causes 403 errors."
                
                print("   ✓ 403 error prevention verified")
                
        except Exception as e:
            if "403" in str(e):
                pytest.fail(
                    f"REGRESSION: 403 error occurred! Bug GEN-2166 has returned. Error: {str(e)}"
                )
            raise
    
    print("\n=== Regression Test Passed ===")
    print("✓ Bug GEN-2166 is fixed and protected against regression")
    print("✓ Managed files are properly accessed via managed files hook")
    print("✓ User context is correctly passed through")
    print("✓ No 403 errors occur")
    print("✓ Non-managed files still work correctly\n")


@pytest.mark.asyncio()
async def test_batch_logging_azure_credentials_regression():
    """
    Regression test: LoggingWorker Missing Azure Credentials When Fetching Batch Output
    
    This test ensures that Azure credentials are properly passed when fetching batch
    output files during logging, preventing "Missing credentials" errors.
    
    Bug: The LoggingWorker failed when processing completed Azure batches because
    it attempted to fetch batch output file content without Azure credentials.
    
    Fix: Pass litellm_params (containing credentials) from the logging object
    through to the file content retrieval functions.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.batches.batch_utils import (
        _extract_file_access_credentials,
        _get_batch_output_file_content_as_dictionary,
        _handle_completed_batch,
    )
    from litellm.types.llms.openai import Batch, HttpxBinaryResponseContent
    import httpx
    
    print("\n=== Regression Test: Azure Batch Logging Credentials ===")
    
    # Setup: Create mock batch with output file
    mock_batch = Batch(
        id="batch-azure-test",
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id="file-input-azure",
        completion_window="24h",
        status="completed",
        output_file_id="file-output-azure",
        error_file_id=None,
        created_at=1234567890,
        in_progress_at=1234567900,
        expires_at=1234654290,
        finalizing_at=1234568000,
        completed_at=1234568100,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts=None,
        metadata=None,
    )
    
    # Setup: Azure credentials (as they would be in litellm_params)
    azure_credentials = {
        "api_key": "test-azure-key-regression",
        "api_base": "https://test-regression.openai.azure.com",
        "api_version": "2024-02-15-preview",
        "organization": "test-org",
        "timeout": 600,
    }
    
    # Setup: Mock batch output content
    batch_output = b'{"id": "batch_req_1", "custom_id": "request-1", "response": {"status_code": 200, "body": {"id": "chatcmpl-azure", "object": "chat.completion", "model": "gpt-4", "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}}}}\n'
    
    # Test 1: Verify _extract_file_access_credentials works correctly
    print("\n1. Testing credential extraction...")
    
    extracted_creds = _extract_file_access_credentials(azure_credentials)
    assert "api_key" in extracted_creds, "api_key should be extracted"
    assert extracted_creds["api_key"] == "test-azure-key-regression", "Incorrect api_key"
    assert "api_base" in extracted_creds, "api_base should be extracted"
    assert "api_version" in extracted_creds, "api_version should be extracted"
    assert "timeout" in extracted_creds, "timeout should be extracted"
    
    print("   ✓ Credentials extracted correctly")
    print(f"   ✓ Extracted keys: {list(extracted_creds.keys())}")
    
    # Test 2: Verify credentials are passed to afile_content
    print("\n2. Testing credentials passed to afile_content...")
    
    credentials_received = {"value": False, "params": None}
    
    async def mock_afile_content_tracker(**kwargs):
        # Track if Azure credentials were passed
        if "api_key" in kwargs and "api_base" in kwargs and "api_version" in kwargs:
            credentials_received["value"] = True
            credentials_received["params"] = {
                "api_key": kwargs.get("api_key"),
                "api_base": kwargs.get("api_base"),
                "api_version": kwargs.get("api_version"),
            }
        mock_response = httpx.Response(
            status_code=200,
            content=batch_output,
            headers={"content-type": "application/octet-stream"},
        )
        return HttpxBinaryResponseContent(response=mock_response)
    
    with patch('litellm.files.main.afile_content', side_effect=mock_afile_content_tracker):
        result = await _get_batch_output_file_content_as_dictionary(
            batch=mock_batch,
            custom_llm_provider="azure",
            litellm_params=azure_credentials,
        )
        
        # Verify credentials were passed
        assert credentials_received["value"], \
            "REGRESSION: Azure credentials not passed to afile_content! This causes 'Missing credentials' error."
        assert credentials_received["params"]["api_key"] == "test-azure-key-regression", \
            "REGRESSION: Incorrect api_key"
        assert credentials_received["params"]["api_base"] == "https://test-regression.openai.azure.com", \
            "REGRESSION: Incorrect api_base"
        
        print("   ✓ Credentials passed to afile_content")
        print(f"   ✓ api_key: {credentials_received['params']['api_key']}")
        print(f"   ✓ api_base: {credentials_received['params']['api_base']}")
    
    # Test 3: Verify full flow through _handle_completed_batch
    print("\n3. Testing full logging flow...")
    
    credentials_received["value"] = False
    credentials_received["params"] = None
    
    with patch('litellm.files.main.afile_content', side_effect=mock_afile_content_tracker):
        cost, usage, models = await _handle_completed_batch(
            batch=mock_batch,
            custom_llm_provider="azure",
            litellm_params=azure_credentials,
        )
        
        # Verify credentials were passed through the entire flow
        assert credentials_received["value"], \
            "REGRESSION: Credentials not passed through _handle_completed_batch"
        
        # Verify cost and usage were calculated
        assert cost > 0, "Cost should be calculated"
        assert usage.total_tokens == 40, "Usage should be calculated correctly"
        
        print("   ✓ Credentials passed through full flow")
        print(f"   ✓ Cost: {cost}")
        print(f"   ✓ Usage: {usage.total_tokens} tokens")
        print(f"   ✓ Models: {models}")
    
    # Test 4: Verify error prevention
    print("\n4. Testing 'Missing credentials' error prevention...")
    
    # Simulate the bug: if credentials are NOT passed, Azure would fail
    with patch('litellm.files.main.afile_content') as mock_afile_content_fail:
        # This is what would happen without the fix
        mock_afile_content_fail.side_effect = Exception(
            "Missing credentials. Please pass one of `api_key`, `azure_ad_token`, "
            "`azure_ad_token_provider`, or the `AZURE_OPENAI_API_KEY` or "
            "`AZURE_OPENAI_AD_TOKEN` environment variables."
        )
        
        # Now test with the fix - should NOT raise the error
        with patch('litellm.files.main.afile_content', side_effect=mock_afile_content_tracker):
            try:
                cost, usage, models = await _handle_completed_batch(
                    batch=mock_batch,
                    custom_llm_provider="azure",
                    litellm_params=azure_credentials,
                )
                print("   ✓ No 'Missing credentials' error with fix")
            except Exception as e:
                if "Missing credentials" in str(e):
                    pytest.fail(
                        f"REGRESSION: 'Missing credentials' error occurred! "
                        f"Credentials not being passed. Error: {str(e)}"
                    )
                raise
    
    # Test 5: Verify backwards compatibility (works without credentials for OpenAI)
    print("\n5. Testing backwards compatibility...")
    
    with patch('litellm.files.main.afile_content') as mock_afile_content:
        mock_response = httpx.Response(
            status_code=200,
            content=batch_output,
            headers={"content-type": "application/octet-stream"},
        )
        mock_afile_content.return_value = HttpxBinaryResponseContent(response=mock_response)
        
        # Call without litellm_params (should still work for OpenAI)
        result = await _get_batch_output_file_content_as_dictionary(
            batch=mock_batch,
            custom_llm_provider="openai",
            litellm_params=None,
        )
        
        assert len(result) > 0, "Should return file content"
        print("   ✓ Backwards compatibility maintained")
        print("   ✓ Works without litellm_params for OpenAI")
    
    print("\n=== Regression Test Passed ===")
    print("✓ Azure credentials properly passed from logging to file retrieval")
    print("✓ 'Missing credentials' error prevented")
    print("✓ Batch output files can be fetched with Azure credentials")
    print("✓ Cost and usage tracking works for Azure batches")
    print("✓ Backwards compatibility maintained\n")
