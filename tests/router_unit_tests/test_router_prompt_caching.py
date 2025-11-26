import sys
import os
import traceback
import asyncio
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from create_mock_standard_logging_payload import create_standard_logging_payload
from litellm.types.utils import StandardLoggingPayload
import unittest
from pydantic import BaseModel
from litellm.router_utils.prompt_caching_cache import PromptCachingCache


class ExampleModel(BaseModel):
    field1: str
    field2: int


def test_serialize_pydantic_object():
    model = ExampleModel(field1="value", field2=42)
    serialized = PromptCachingCache.serialize_object(model)
    assert serialized == {"field1": "value", "field2": 42}


def test_serialize_dict():
    obj = {"b": 2, "a": 1}
    serialized = PromptCachingCache.serialize_object(obj)
    assert serialized == '{"a":1,"b":2}'  # JSON string with sorted keys


def test_serialize_nested_dict():
    obj = {"z": {"b": 2, "a": 1}, "x": [1, 2, {"c": 3}]}
    serialized = PromptCachingCache.serialize_object(obj)
    expected = '{"x":[1,2,{"c":3}],"z":{"a":1,"b":2}}'  # JSON string with sorted keys
    assert serialized == expected


def test_serialize_list():
    obj = ["item1", {"a": 1, "b": 2}, 42]
    serialized = PromptCachingCache.serialize_object(obj)
    expected = ["item1", '{"a":1,"b":2}', 42]
    assert serialized == expected


def test_serialize_fallback():
    obj = 12345  # Simple non-serializable object
    serialized = PromptCachingCache.serialize_object(obj)
    assert serialized == 12345


def test_serialize_non_serializable():
    class CustomClass:
        def __str__(self):
            return "custom_object"

    obj = CustomClass()
    serialized = PromptCachingCache.serialize_object(obj)
    assert serialized == "custom_object"  # Fallback to string conversion


@pytest.mark.asyncio
async def test_router_prompt_caching_same_cacheable_prefix_routes_to_same_deployment():
    """
    End-to-end test to validate prompt caching routing through LiteLLM Router.
    
    Tests that requests with same cacheable content but different user messages
    route to the same deployment (for prompt caching).
    
    This reproduces the issue where requests with same cacheable prefix but different
    user messages should route to the same deployment, but previously didn't because
    the cache key included the entire messages array instead of just the cacheable prefix.
    """
    from litellm.types.llms.openai import AllMessageValues
    
    def create_messages(user_content: str) -> list[AllMessageValues]:
        """
        Create messages matching the user's exact scenario.
        
        Message structure:
        - BLOCK 1: System message, first content block (no cache_control)
                  → INCLUDED (comes before the last cacheable block)
        - BLOCK 2: System message, second content block (WITH cache_control)
                  → INCLUDED (this IS the last cacheable block)
        - USER MESSAGE: User message (no cache_control)
                  → NOT included (comes after last cacheable block)
        """
        return [
            {
                "role": "system",
                "content": [
                    # BLOCK 1: No cache_control → INCLUDED (all blocks up to last cacheable are included)
                    {"type": "text", "text": "You are an AI assistant tasked with analyzing legal documents."},
                    # BLOCK 2: Has cache_control → INCLUDED (this is the last cacheable block)
                    {
                        "type": "text",
                        "text": "Here 3 is the full text of a complex legal agreement" * 400,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {
                "role": "user",
                # USER MESSAGE: No cache_control → NOT included (comes after last cacheable block)
                "content": user_content,
            },
        ]
    
    # Create router with multiple deployments
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_base": "https://exampleopenaiendpoint-production-0ee2.up.railway.app/v1",
                    "api_key": f"test-key-{i}",
                },
                "model_info": {"id": f"deployment-{i}"},
            }
            for i in range(1, 4)
        ],
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["prompt_caching"],
    )
    
    # Create test messages matching user's exact scenario
    # Same cacheable prefix (system blocks 1+2) but different user messages
    messages1 = create_messages("what are the key terms and conditions in this agreement?")
    messages2 = create_messages("how many words are there?")
    messages3 = create_messages("how many sentences are there?")
    
    cache = PromptCachingCache(cache=router.cache)
    
    # Test 1: Cache keys should be same (same cacheable prefix, different user messages)
    key1 = PromptCachingCache.get_prompt_caching_cache_key(messages1, None)
    key2 = PromptCachingCache.get_prompt_caching_cache_key(messages2, None)
    key3 = PromptCachingCache.get_prompt_caching_cache_key(messages3, None)
    
    assert key1 is not None, "Cache key should not be None"
    assert key1 == key2 == key3, "Cache keys should be the same for same cacheable prefix"
    
    # Make first request
    try:
        response1 = await router.acompletion(model="test-model", messages=messages1)
        model_id_1 = response1._hidden_params.get("model_id", "unknown")
    except Exception:
        # If API call fails, we can still test the cache key logic
        model_id_1 = "unknown"
    
    await asyncio.sleep(1)  # Wait for cache write
    
    # Test 2: Cache lookup should work for messages2 (same cacheable prefix)
    cached_2 = await cache.async_get_model_id(messages2, None)
    # Cache should be found if first request succeeded
    if model_id_1 != "unknown":
        assert cached_2 is not None, "Cache lookup should work for same cacheable prefix"
    
    # Make second request
    try:
        response2 = await router.acompletion(model="test-model", messages=messages2)
        model_id_2 = response2._hidden_params.get("model_id", "unknown")
    except Exception:
        model_id_2 = "unknown"
    
    await asyncio.sleep(1)  # Wait for cache write
    
    # Make third request
    try:
        response3 = await router.acompletion(model="test-model", messages=messages3)
        model_id_3 = response3._hidden_params.get("model_id", "unknown")
    except Exception:
        model_id_3 = "unknown"
    
    # Test 3: All requests should route to same deployment (if API calls succeeded)
    if model_id_1 != "unknown" and model_id_2 != "unknown" and model_id_3 != "unknown":
        assert (
            model_id_1 == model_id_2 == model_id_3
        ), f"All requests should route to same deployment, but got: {model_id_1}, {model_id_2}, {model_id_3}"
