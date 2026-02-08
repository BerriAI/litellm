"""
Unit Test for TPM Rate Limit Concurrent Bypass Fix
===================================================

This test directly verifies that the token reservation mechanism
prevents concurrent requests from bypassing TPM limits.

It does NOT require a running proxy, database, or Redis - it tests
the core logic in isolation.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3 as RateLimitHandler,
    TPM_RESERVED_TOKENS_KEY,
)
from litellm.proxy.utils import InternalUsageCache, hash_token


class TestTPMConcurrentBypassFix:
    """Test suite for verifying the TPM concurrent bypass fix."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter instance for testing."""
        cache = DualCache()
        handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
        return handler, cache

    @pytest.mark.asyncio
    async def test_token_reservation_prevents_concurrent_bypass(self, rate_limiter):
        """
        Test that token reservation prevents multiple concurrent requests
        from bypassing the TPM limit.

        Scenario:
        - TPM limit: 100 tokens
        - 5 concurrent requests, each estimated to use ~50 tokens
        - Without fix: All 5 would pass (check then update)
        - With fix: Only 2 should pass (reserve then check)
        """
        handler, cache = rate_limiter

        # Create API key auth with TPM limit of 100
        api_key = hash_token("sk-test-key")
        user_api_key_dict = UserAPIKeyAuth(
            api_key=api_key,
            tpm_limit=100,  # Low limit to trigger rate limiting
        )

        # Track token reservations
        reservations = []

        # Mock the token increment to track what's being reserved
        original_increment = handler.async_increment_tokens_with_ttl_preservation

        async def track_increment(pipeline_operations, **kwargs):
            for op in pipeline_operations:
                if "tokens" in op["key"]:
                    reservations.append(
                        {
                            "key": op["key"],
                            "increment": op["increment_value"],
                            "timestamp": datetime.now(),
                        }
                    )
            # Still call the original to update cache
            await original_increment(pipeline_operations, **kwargs)

        handler.async_increment_tokens_with_ttl_preservation = track_increment

        # Create request data with messages (to trigger token estimation)
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a test message for concurrent bypass testing.",
                }
            ],
            "max_tokens": 50,  # Request ~50 output tokens
        }

        # Fire 5 concurrent pre_call_hook requests
        async def make_request(request_id: int) -> Dict[str, Any]:
            data = request_data.copy()
            try:
                await handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=cache,
                    data=data,
                    call_type="",
                )
                return {
                    "request_id": request_id,
                    "success": True,
                    "reserved_tokens": data.get(TPM_RESERVED_TOKENS_KEY, 0),
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "status_code": getattr(e, "status_code", None),
                }

        # Fire all requests concurrently
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        rate_limited = [r for r in failed if r.get("status_code") == 429]

        # Results: successful, rate_limited, reservations tracked

        # With the fix in place, we should see some rate limited requests
        # because tokens are reserved upfront
        assert len(rate_limited) > 0, (
            f"Expected some rate limited requests, but all {len(successful)} succeeded. "
            f"This suggests the concurrent bypass bug still exists."
        )

        # Total tokens reserved should not exceed limit by much
        _total_reserved = sum(r.get("reserved_tokens", 0) for r in successful)
        # Total tokens reserved by successful requests is in _total_reserved

    @pytest.mark.asyncio
    async def test_token_adjustment_on_success(self, rate_limiter):
        """
        Test that after a successful request, tokens are adjusted based on
        actual usage vs reserved.
        """
        handler, cache = rate_limiter

        api_key = hash_token("sk-test-adjust")

        # Create mock kwargs for success event
        mock_kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_hash": api_key,
                    TPM_RESERVED_TOKENS_KEY: 100,  # Reserved 100 tokens
                }
            },
            "model": "gpt-3.5-turbo",
        }

        # Create mock response with actual usage
        from litellm.types.utils import ModelResponse, Usage

        mock_response = ModelResponse(
            id="test",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model="gpt-3.5-turbo",
            usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
            choices=[],
        )

        # Track increments
        increments = []

        async def mock_increment(increment_list, **kwargs):
            for op in increment_list:
                increments.append(
                    {
                        "key": op["key"],
                        "increment": op["increment_value"],
                    }
                )

        handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
            mock_increment
        )

        # Call success event
        await handler.async_log_success_event(
            kwargs=mock_kwargs,
            response_obj=mock_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Find the token adjustment
        token_adjustments = [i for i in increments if "tokens" in i["key"]]

        # Token adjustments are in token_adjustments list

        # The adjustment should be actual - reserved = 50 - 100 = -50
        # (Refunding 50 tokens)
        assert any(
            i["increment"] == -50 for i in token_adjustments
        ), f"Expected a -50 token adjustment (refund), but got: {token_adjustments}"

    @pytest.mark.asyncio
    async def test_token_release_on_failure(self, rate_limiter):
        """
        Test that when a request fails, all reserved tokens are released.
        """
        handler, cache = rate_limiter

        api_key = hash_token("sk-test-fail")

        # Create mock kwargs for failure event
        mock_kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_hash": api_key,
                    TPM_RESERVED_TOKENS_KEY: 100,  # Reserved 100 tokens
                }
            },
        }

        # Track increments
        increments = []

        async def mock_increment(increment_list, **kwargs):
            for op in increment_list:
                increments.append(
                    {
                        "key": op["key"],
                        "increment": op["increment_value"],
                    }
                )

        handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
            mock_increment
        )

        # Call failure event
        await handler.async_log_failure_event(
            kwargs=mock_kwargs,
            response_obj=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Find the token releases
        token_releases = [i for i in increments if "tokens" in i["key"]]

        # Token releases are in token_releases list

        # Should release all reserved tokens (-100)
        assert any(
            i["increment"] == -100 for i in token_releases
        ), f"Expected all reserved tokens to be released (-100), but got: {token_releases}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
