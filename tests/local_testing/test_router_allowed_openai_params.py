"""
Test for issue #15845 - allowed_openai_params in config.yaml not working

This test ensures that when allowed_openai_params is set in the litellm_params
of a model config, it is correctly passed through to the completion call.
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from unittest.mock import patch, MagicMock
import pytest


def test_router_allowed_openai_params_from_config_sync():
    """
    Test that allowed_openai_params from litellm_params in config
    is respected when calling router.completion()
    
    Regression test for: https://github.com/BerriAI/litellm/issues/15845
    """
    # Setup: Create router with allowed_openai_params in config
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "api_key": "fake-key",
                "allowed_openai_params": ["tools", "response_format"]
            }
        }
    ]
    
    router = Router(model_list=model_list)
    
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]
    
    # Mock the actual completion call to check what parameters are passed
    with patch('litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock()
        
        try:
            router.completion(
                model="test-model",
                messages=messages,
                tools=tools
            )
        except Exception:
            # Ignore errors from mocking
            pass
        
        # Check that the completion was called with allowed_openai_params
        assert mock_completion.called, "litellm.completion should have been called"
        call_kwargs = mock_completion.call_args[1]
        
        # Verify that allowed_openai_params was passed
        assert "allowed_openai_params" in call_kwargs, \
            "allowed_openai_params should be in the call kwargs"
        assert call_kwargs["allowed_openai_params"] == ["tools", "response_format"], \
            f"allowed_openai_params should be ['tools', 'response_format'], got {call_kwargs['allowed_openai_params']}"


def test_router_allowed_openai_params_override():
    """
    Test that allowed_openai_params passed in request overrides config value
    
    This ensures that users can still override the config value on a per-request basis.
    """
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "api_key": "fake-key",
                "allowed_openai_params": ["tools"]
            }
        }
    ]
    
    router = Router(model_list=model_list)
    
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch('litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock()
        
        try:
            # Override with different value in request
            router.completion(
                model="test-model",
                messages=messages,
                allowed_openai_params=["response_format"]
            )
        except Exception:
            pass
        
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]
        
        # Verify that request override worked
        assert call_kwargs["allowed_openai_params"] == ["response_format"], \
            f"allowed_openai_params should be overridden to ['response_format'], got {call_kwargs['allowed_openai_params']}"


def test_router_allowed_openai_params_not_set():
    """
    Test that when allowed_openai_params is not set in config,
    it doesn't interfere with normal operation
    """
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "api_key": "fake-key",
                # Note: no allowed_openai_params
            }
        }
    ]
    
    router = Router(model_list=model_list)
    
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch('litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock()
        
        try:
            router.completion(
                model="test-model",
                messages=messages
            )
        except Exception:
            pass
        
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]
        
        # When not set in config and not in request, it should be None or not present
        # (depending on how kwargs are processed)
        allowed_params = call_kwargs.get("allowed_openai_params")
        assert allowed_params is None or allowed_params == [], \
            f"allowed_openai_params should be None or empty, got {allowed_params}"


@pytest.mark.asyncio
async def test_router_allowed_openai_params_from_config_async():
    """
    Test that allowed_openai_params from litellm_params in config
    is respected when calling router.acompletion()
    
    Regression test for: https://github.com/BerriAI/litellm/issues/15845
    """
    # Setup: Create router with allowed_openai_params in config
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "api_key": "fake-key",
                "allowed_openai_params": ["tools", "response_format"]
            }
        }
    ]
    
    router = Router(model_list=model_list)
    
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]
    
    # Mock the actual completion call to check what parameters are passed
    with patch('litellm.acompletion') as mock_completion:
        # Create a coroutine that returns a mock
        async def mock_acompletion(*args, **kwargs):
            return MagicMock()
        
        mock_completion.side_effect = mock_acompletion
        
        try:
            await router.acompletion(
                model="test-model",
                messages=messages,
                tools=tools
            )
        except Exception:
            # Ignore errors from mocking
            pass
        
        # Check that the completion was called with allowed_openai_params
        assert mock_completion.called, "litellm.acompletion should have been called"
        call_kwargs = mock_completion.call_args[1]
        
        # Verify that allowed_openai_params was passed
        assert "allowed_openai_params" in call_kwargs, \
            "allowed_openai_params should be in the call kwargs"
        assert call_kwargs["allowed_openai_params"] == ["tools", "response_format"], \
            f"allowed_openai_params should be ['tools', 'response_format'], got {call_kwargs['allowed_openai_params']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
