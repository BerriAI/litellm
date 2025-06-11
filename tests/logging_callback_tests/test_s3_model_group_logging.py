import os
import sys
import traceback
import uuid
import pytest
from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import os
import time
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
import asyncio
from typing import Optional
from litellm.types.utils import StandardLoggingPayload, Usage, ModelInfoBase
from litellm.integrations.custom_logger import CustomLogger


class TestS3Logger(CustomLogger):
    def __init__(self):
        self.recorded_usage: Optional[Usage] = None
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object")
        self.standard_logging_payload = standard_logging_payload
        print(
            "standard_logging_payload",
            json.dumps(standard_logging_payload, indent=4, default=str),
        )


@pytest.mark.asyncio
async def test_moderation_model_group_logging():
    """
    Test that moderation requests properly set model_group in standard_logging_payload
    This test reproduces the issue where model_group="" for moderation requests
    """
    custom_logger = TestS3Logger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    input_content = "Hello, how are you?"
    model = "text-moderation-stable"
    
    response = await litellm.amoderation(
        input=input_content,
        model=model,
    )

    print("response", json.dumps(response, indent=4, default=str))

    await asyncio.sleep(2)

    assert custom_logger.standard_logging_payload is not None

    # validate the standard_logging_payload
    standard_logging_payload: StandardLoggingPayload = custom_logger.standard_logging_payload
    assert standard_logging_payload["call_type"] == litellm.utils.CallTypes.amoderation.value
    assert standard_logging_payload["status"] == "success"
    
    # This is the main test - model_group should be set to the model name
    assert standard_logging_payload["model_group"] is not None
    assert standard_logging_payload["model_group"] != ""
    assert standard_logging_payload["model_group"] == model

    print(f"✅ model_group is correctly set to: {standard_logging_payload['model_group']}")


@pytest.mark.asyncio
async def test_anthropic_messages_model_group_logging():
    """
    Test that anthropic_messages requests properly set model_group in standard_logging_payload
    This test reproduces the issue where model_group="" for anthropic_messages requests
    """
    custom_logger = TestS3Logger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    model = "claude-3-haiku-20240307"
    
    try:
        response = await litellm.aanthropic_messages(
            model=model,
            messages=messages,
            max_tokens=10,
        )

        print("response", json.dumps(response, indent=4, default=str))

        await asyncio.sleep(2)

        assert custom_logger.standard_logging_payload is not None

        # validate the standard_logging_payload
        standard_logging_payload: StandardLoggingPayload = custom_logger.standard_logging_payload
        assert standard_logging_payload["call_type"] == litellm.utils.CallTypes.anthropic_messages.value
        assert standard_logging_payload["status"] == "success"
        
        # This is the main test - model_group should be set to the model name
        assert standard_logging_payload["model_group"] is not None
        assert standard_logging_payload["model_group"] != ""
        assert standard_logging_payload["model_group"] == model

        print(f"✅ model_group is correctly set to: {standard_logging_payload['model_group']}")
    
    except Exception as e:
        # Skip test if anthropic API key is not available
        if "api key" in str(e).lower() or "authentication" in str(e).lower():
            pytest.skip(f"Skipping anthropic_messages test due to missing API key: {e}")
        else:
            raise


@pytest.mark.asyncio 
async def test_router_moderation_model_group_logging():
    """
    Test that router moderation requests properly set model_group in standard_logging_payload
    This tests the router's moderation endpoint specifically
    """
    from litellm import Router
    
    # Create a router with a moderation model
    model_list = [
        {
            "model_name": "text-moderation-stable",
            "litellm_params": {
                "model": "text-moderation-stable",
            },
        }
    ]
    
    router = Router(model_list=model_list)
    
    custom_logger = TestS3Logger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    input_content = "Hello, how are you?"
    model = "text-moderation-stable"
    
    response = await router.amoderation(
        input=input_content,
        model=model,
    )

    print("response", json.dumps(response, indent=4, default=str))

    await asyncio.sleep(2)

    assert custom_logger.standard_logging_payload is not None

    # validate the standard_logging_payload
    standard_logging_payload: StandardLoggingPayload = custom_logger.standard_logging_payload
    assert standard_logging_payload["call_type"] == litellm.utils.CallTypes.amoderation.value
    assert standard_logging_payload["status"] == "success"
    
    # This is the main test - model_group should be set to the model name
    assert standard_logging_payload["model_group"] is not None
    assert standard_logging_payload["model_group"] != ""
    assert standard_logging_payload["model_group"] == model

    print(f"✅ Router moderation model_group is correctly set to: {standard_logging_payload['model_group']}")


@pytest.mark.asyncio
async def test_router_anthropic_messages_model_group_logging():
    """
    Test that router anthropic_messages requests properly set model_group in standard_logging_payload
    This tests the router's anthropic_messages endpoint specifically
    """
    from litellm import Router
    
    # Create a router with an anthropic model
    model_list = [
        {
            "model_name": "claude-3-haiku-20240307",
            "litellm_params": {
                "model": "claude-3-haiku-20240307",
            },
        }
    ]
    
    router = Router(model_list=model_list)
    
    custom_logger = TestS3Logger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    model = "claude-3-haiku-20240307"
    
    try:
        response = await router.aanthropic_messages(
            model=model,
            messages=messages,
            max_tokens=10,
        )

        print("response", json.dumps(response, indent=4, default=str))

        await asyncio.sleep(2)

        assert custom_logger.standard_logging_payload is not None

        # validate the standard_logging_payload
        standard_logging_payload: StandardLoggingPayload = custom_logger.standard_logging_payload
        assert standard_logging_payload["call_type"] == litellm.utils.CallTypes.anthropic_messages.value
        assert standard_logging_payload["status"] == "success"
        
        # This is the main test - model_group should be set to the model name
        assert standard_logging_payload["model_group"] is not None
        assert standard_logging_payload["model_group"] != ""
        assert standard_logging_payload["model_group"] == model

        print(f"✅ Router anthropic_messages model_group is correctly set to: {standard_logging_payload['model_group']}")
    
    except Exception as e:
        # Skip test if anthropic API key is not available
        if "api key" in str(e).lower() or "authentication" in str(e).lower():
            pytest.skip(f"Skipping router anthropic_messages test due to missing API key: {e}")
        else:
            raise