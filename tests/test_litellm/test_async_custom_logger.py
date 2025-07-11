### Test async custom logger callbacks - Issue #8842
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger


class TestAsyncCallbackHandler(CustomLogger):
    """Handler for testing async callbacks - Issue #8842"""
    def __init__(self):
        super().__init__()
        self.sync_success_called = False
        self.async_success_called = False
        self.sync_failure_called = False
        self.async_failure_called = False
        self.pre_api_call_called = False
        self.post_api_call_called = False

    def log_pre_api_call(self, model, messages, kwargs):
        self.pre_api_call_called = True

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        self.post_api_call_called = True

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.sync_success_called = True

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.sync_failure_called = True

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.async_success_called = True

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.async_failure_called = True


@pytest.mark.asyncio
async def test_router_acompletion_triggers_async_callbacks():
    """Test that router.acompletion() properly triggers async callbacks - Issue #8842"""
    handler = TestAsyncCallbackHandler()
    
    # Save existing callbacks and set our handler
    original_callbacks = litellm.callbacks
    litellm.callbacks = [handler]
    
    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "Test response"},
                },
            ],
        )
        
        # Make async completion call
        response = await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "Test message"}],
        )
        
        # Wait for async callbacks to complete
        await asyncio.sleep(0.1)
        
        # Verify callbacks were called
        assert handler.pre_api_call_called, "Pre-API call callback should be called"
        assert handler.post_api_call_called, "Post-API call callback should be called"
        # Note: For CustomLogger instances in async contexts, only async callbacks are triggered
        # Sync callbacks are filtered out by design to avoid duplicate processing
        assert handler.async_success_called, "Async success callback should be called"
        assert not handler.sync_success_called, "Sync success callback should NOT be called for CustomLogger in async context"
        
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_direct_acompletion_triggers_async_callbacks():
    """Test that direct litellm.acompletion() properly triggers async callbacks - Issue #8842"""
    handler = TestAsyncCallbackHandler()
    
    # Save existing callbacks and set our handler
    original_callbacks = litellm.callbacks
    litellm.callbacks = [handler]
    
    try:
        # Make async completion call
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            mock_response="Test response",
        )
        
        # Wait for async callbacks to complete
        await asyncio.sleep(0.1)
        
        # Verify callbacks were called
        assert handler.pre_api_call_called, "Pre-API call callback should be called"
        assert handler.post_api_call_called, "Post-API call callback should be called"
        # Note: For CustomLogger instances in async contexts, only async callbacks are triggered
        # Sync callbacks are filtered out by design to avoid duplicate processing
        assert handler.async_success_called, "Async success callback should be called"
        assert not handler.sync_success_called, "Sync success callback should NOT be called for CustomLogger in async context"
        
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_sync_completion_triggers_sync_callbacks():
    """Test that sync completion() triggers sync callbacks but not async - Issue #8842"""
    handler = TestAsyncCallbackHandler()
    
    # Save existing callbacks and set our handler
    original_callbacks = litellm.callbacks
    litellm.callbacks = [handler]
    
    try:
        # Make sync completion call
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            mock_response="Test response",
        )
        
        # Wait a bit to ensure no async callbacks are triggered
        await asyncio.sleep(0.1)
        
        # Verify sync callbacks were called but not async
        assert handler.pre_api_call_called, "Pre-API call callback should be called"
        assert handler.post_api_call_called, "Post-API call callback should be called"
        assert handler.sync_success_called, "Sync success callback should be called"
        assert not handler.async_success_called, "Async success callback should NOT be called for sync completion"
        
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks