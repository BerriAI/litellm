"""
Integration tests for async_post_call_failure_hook.

Tests verify that the failure hook can transform error responses sent to clients,
similar to how async_post_call_success_hook can transform successful responses.
"""

import os
import sys
import pytest
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../../.."))

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class ErrorTransformerLogger(CustomLogger):
    """Logger that transforms errors into user-friendly messages"""
    
    def __init__(self):
        self.called = False
        self.transformed_exception = None
    
    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ):
        self.called = True
        self.transformed_exception = HTTPException(
            status_code=400,
            detail="User-friendly error: Your request could not be processed."
        )
        return self.transformed_exception


@pytest.mark.asyncio
async def test_failure_hook_transforms_error_response():
    """
    Test that async_post_call_failure_hook can transform error responses.
    This mirrors how async_post_call_success_hook can transform successful responses.
    """
    transformer = ErrorTransformerLogger()
    
    # Mock litellm.callbacks to include our transformer
    with patch("litellm.callbacks", [transformer]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache
        
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        original_exception = Exception("Technical error message")
        request_data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
        
        # Call the hook
        result = await proxy_logging.post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )
        
        # Verify hook was called
        assert transformer.called is True
        
        # Verify transformed exception is returned
        assert result is not None
        assert isinstance(result, HTTPException)
        assert result.detail == "User-friendly error: Your request could not be processed."


@pytest.mark.asyncio
async def test_failure_hook_returns_none_when_no_transformation():
    """
    Test that hook returning None uses original exception.
    """
    class NoOpLogger(CustomLogger):
        def __init__(self):
            self.called = False
        
        async def async_post_call_failure_hook(self, *args, **kwargs):
            self.called = True
            return None
    
    logger = NoOpLogger()
    
    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache
        
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        original_exception = Exception("Original error")
        request_data = {"model": "test"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test")
        
        result = await proxy_logging.post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )
        
        # Should return None (original exception will be used)
        assert result is None
        assert logger.called is True


@pytest.mark.asyncio
async def test_failure_hook_handles_exceptions_gracefully():
    """
    Test that hook failures don't break the error flow.
    """
    class FailingLogger(CustomLogger):
        def __init__(self):
            self.called = False
        
        async def async_post_call_failure_hook(self, *args, **kwargs):
            self.called = True
            raise RuntimeError("Hook crashed!")
    
    logger = FailingLogger()
    
    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache
        
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        original_exception = Exception("Original error")
        request_data = {"model": "test"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test")
        
        # Should not raise, should handle gracefully
        result = await proxy_logging.post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )
        
        # Should return None (original exception will be used)
        assert result is None
        assert logger.called is True

