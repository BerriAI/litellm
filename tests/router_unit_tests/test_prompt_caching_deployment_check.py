import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
)
from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.caching.dual_cache import DualCache
from litellm.types.utils import CallTypes
from create_mock_standard_logging_payload import create_standard_logging_payload


@pytest.mark.asyncio
async def test_anthropic_messages_call_type_is_cached():
    """
    Regression test: Verify that anthropic_messages call type is allowed
    in PromptCachingDeploymentCheck.async_log_success_event.
    """
    cache = DualCache()
    deployment_check = PromptCachingDeploymentCheck(cache=cache)
    prompt_cache = PromptCachingCache(cache=cache)
    
    # Create messages with enough tokens to pass the caching threshold
    test_messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": "test long message here" * 1024,
                    "cache_control": {
                        "type": "ephemeral",
                        "ttl": "5m"
                    }
                }
            ]
        }
    ]
    test_model_id = "test-model-id-123"
    
    # Create a payload with anthropic_messages call type
    payload = create_standard_logging_payload()
    payload["call_type"] = CallTypes.anthropic_messages.value
    payload["messages"] = test_messages
    payload["model"] = "anthropic/claude-3-5-sonnet-20240620"
    payload["model_id"] = test_model_id
    
    # Log the success event (should cache the model_id)
    await deployment_check.async_log_success_event(
        kwargs={"standard_logging_object": payload},
        response_obj={},
        start_time=1234567890.0,
        end_time=1234567891.0,
    )
    
    # Small delay to ensure cache write completes
    await asyncio.sleep(0.1)
    
    # Verify that the model_id was actually cached
    cached_result = await prompt_cache.async_get_model_id(
        messages=test_messages,
        tools=None,
    )
    
    # This assertion will FAIL if anthropic_messages is filtered out
    assert cached_result is not None, "Model ID should be cached for anthropic_messages call type"
    assert cached_result["model_id"] == test_model_id, f"Expected {test_model_id}, got {cached_result['model_id']}"