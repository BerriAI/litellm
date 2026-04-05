"""
Test that dynamic_rate_limiter_v3 raises litellm.RateLimitError instead of HTTPException
to enable fallback routing.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3 as DynamicRateLimitHandler,
)


@pytest.mark.asyncio
async def test_rate_limiter_raises_litellm_rate_limit_error():
    """
    Test that dynamic_rate_limiter_v3 raises litellm.RateLimitError instead of HTTPException.
    
    This is critical for fallback behavior - the router's fallback logic catches
    litellm.RateLimitError, not fastapi.HTTPException.
    
    Regression test for: https://github.com/BerriAI/litellm/issues/23749
    """
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    litellm.priority_reservation = {"high": 0.7, "low": 0.3}
    
    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)
    
    model = "test-fallback-model"
    total_rpm = 10  # Very low RPM to trigger rate limit
    
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)
    
    # Create user
    user = UserAPIKeyAuth()
    user.metadata = {"priority": "low"}
    user.user_id = "test_user"
    
    # Mock the v3 limiter to return OVER_LIMIT
    async def mock_should_rate_limit(descriptors, parent_otel_span=None, read_only=False):
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "model_saturation_check",
                    "rate_limit_type": "requests",
                    "limit_remaining": 0,
                }
            ],
        }
    
    handler.v3_limiter.should_rate_limit = mock_should_rate_limit
    
    # Test that litellm.RateLimitError is raised (not HTTPException)
    with pytest.raises(litellm.RateLimitError) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user,
            cache=dual_cache,
            data={"model": model},
            call_type="completion",
        )
    
    # Verify the error message contains expected content
    error_message = str(exc_info.value)
    assert "Model capacity reached" in error_message or "rate limit" in error_message.lower()
    assert model in error_message


@pytest.mark.asyncio
async def test_rate_limiter_raises_litellm_rate_limit_error_for_priority():
    """
    Test that priority-based rate limits also raise litellm.RateLimitError.
    
    This tests the case where priority limits are enforced (saturation >= threshold).
    """
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    litellm.priority_reservation = {"high": 0.7, "low": 0.3}
    
    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)
    
    model = "test-fallback-model-priority"
    total_rpm = 100
    
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": total_rpm,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)
    
    # Create user
    user = UserAPIKeyAuth()
    user.metadata = {"priority": "low"}
    user.user_id = "test_user"
    
    # Mock the v3 limiter to return OVER_LIMIT for priority_model
    async def mock_should_rate_limit(descriptors, parent_otel_span=None, read_only=False):
        # Check if priority_model descriptor is present
        has_priority_descriptor = any(
            d.get("key") == "priority_model" for d in descriptors
        )
        
        if has_priority_descriptor and read_only:
            return {
                "overall_code": "OVER_LIMIT",
                "statuses": [
                    {
                        "code": "OVER_LIMIT",
                        "descriptor_key": "priority_model",
                        "rate_limit_type": "requests",
                        "limit_remaining": 0,
                    }
                ],
            }
        
        # For non-read-only or model_saturation_check, return OK
        return {
            "overall_code": "OK",
            "statuses": [
                {
                    "code": "OK",
                    "descriptor_key": "model_saturation_check",
                    "rate_limit_type": "requests",
                    "limit_remaining": 100,
                }
            ],
        }
    
    # Mock saturation check to return high saturation (above threshold)
    async def mock_get_saturation(*args, **kwargs):
        return 0.9  # 90% saturation, above default 80% threshold
    
    handler._check_model_saturation = mock_get_saturation
    handler.v3_limiter.should_rate_limit = mock_should_rate_limit
    
    # Test that litellm.RateLimitError is raised for priority-based limit
    with pytest.raises(litellm.RateLimitError) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user,
            cache=dual_cache,
            data={"model": model},
            call_type="completion",
        )
    
    # Verify the error message
    error_message = str(exc_info.value)
    assert "Priority-based rate limit" in error_message or "rate limit" in error_message.lower()


@pytest.mark.asyncio
async def test_rate_limit_error_not_http_exception():
    """
    Verify that the exception raised is NOT an HTTPException from fastapi.
    
    This ensures the fix correctly addresses issue #23749.
    """
    from fastapi import HTTPException
    
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    litellm.priority_reservation = {"high": 0.7, "low": 0.3}
    
    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)
    
    model = "test-not-http-exception"
    
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": 10,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)
    
    user = UserAPIKeyAuth()
    user.metadata = {"priority": "low"}
    user.user_id = "test_user"
    
    async def mock_should_rate_limit(descriptors, parent_otel_span=None, read_only=False):
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "model_saturation_check",
                    "rate_limit_type": "requests",
                    "limit_remaining": 0,
                }
            ],
        }
    
    handler.v3_limiter.should_rate_limit = mock_should_rate_limit
    
    # Test that HTTPException is NOT raised
    with pytest.raises(Exception) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user,
            cache=dual_cache,
            data={"model": model},
            call_type="completion",
        )
    
    # Verify it's NOT an HTTPException
    assert not isinstance(exc_info.value, HTTPException), (
        f"Expected litellm.RateLimitError but got HTTPException: {exc_info.value}"
    )
    
    # Verify it IS a litellm.RateLimitError
    assert isinstance(exc_info.value, litellm.RateLimitError), (
        f"Expected litellm.RateLimitError but got {type(exc_info.value)}: {exc_info.value}"
    )
