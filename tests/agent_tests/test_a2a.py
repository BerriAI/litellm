"""
Test for LiteLLM A2A module.

Run with:
    pytest tests/agent_tests/test_a2a.py -v -s
"""

import asyncio
import os
import sys
import json
from typing import Optional
from uuid import uuid4

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from a2a.types import MessageSendParams, SendMessageRequest


@pytest.mark.asyncio
async def test_asend_message_with_client_decorator():
    """
    Test asend_message standalone function with @client decorator.
    This tests the LiteLLM logging integration.
    """
    litellm._turn_on_debug()
    from litellm.a2a_protocol import asend_message, create_a2a_client

    # Create the A2A client first
    a2a_client = await create_a2a_client(base_url="http://localhost:10001")

    # Build the request matching A2A SDK spec
    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Hello from @client decorated function!",
                }
            ],
            "messageId": uuid4().hex,
        },
    }

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload),
    )

    # Send message using standalone function with @client decorator
    response = await asend_message(a2a_client=a2a_client, request=request)

    # Print response for debugging
    print("\n=== A2A Response (standalone with @client) ===")
    print(response.model_dump(mode="json", exclude_none=True))

    # Basic assertions
    assert response is not None


class TestA2ALogger(CustomLogger):
    """Custom logger to capture A2A logging payloads for testing."""

    def __init__(self):
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None
        self.logged_kwargs: Optional[dict] = None
        self.log_success_called = False
        super().__init__()

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        print("TestA2ALogger: async_log_success_event called")
        self.log_success_called = True
        self.logged_kwargs = kwargs
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        print(f"Captured standard_logging_payload: {self.standard_logging_payload}")


@pytest.mark.asyncio
async def test_a2a_logging_payload():
    """
    Test that A2A calls create a standard logging payload.
    Validates the @client decorator integration with LiteLLM logging.
    """
    # Reset callbacks and set up custom logger
    litellm.logging_callback_manager._reset_all_callbacks()
    test_logger = TestA2ALogger()
    litellm.callbacks = [test_logger]

    from litellm.a2a_protocol import asend_message, create_a2a_client

    # Create the A2A client first
    a2a_client = await create_a2a_client(base_url="http://localhost:10001")

    # Build the request
    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Hello! Testing logging payload.",
                }
            ],
            "messageId": uuid4().hex,
        },
    }

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload),
    )

    # Send message
    response = await asend_message(a2a_client=a2a_client, request=request)

    # Give async logging time to complete
    await asyncio.sleep(1)

    # Print debug info
    print("\n=== Logging Validation ===")
    print(f"log_success_called: {test_logger.log_success_called}")
    print(f"standard_logging_payload: {test_logger.standard_logging_payload}")
    print(f"logged kwargs: {json.dumps(test_logger.logged_kwargs, indent=4, default=str)}")

    # Verify logging was called
    assert test_logger.log_success_called is True
    assert test_logger.standard_logging_payload is not None

    # Verify standard_logging_payload exists
    slp = test_logger.standard_logging_payload
    assert slp is not None

    # Get values from standard logging payload
    logged_model = slp.get("model") if isinstance(slp, dict) else getattr(slp, "model", None)
    logged_provider = slp.get("custom_llm_provider") if isinstance(slp, dict) else getattr(slp, "custom_llm_provider", None)
    call_type = slp.get("call_type") if isinstance(slp, dict) else getattr(slp, "call_type", None)
    response_cost = slp.get("response_cost") if isinstance(slp, dict) else getattr(slp, "response_cost", None)

    print(f"\n=== Standard Logging Payload Validation ===")
    print(f"model: {logged_model}")
    print(f"custom_llm_provider: {logged_provider}")
    print(f"call_type: {call_type}")
    print(f"response_cost: {response_cost}")

    # Verify model and custom_llm_provider are set correctly
    assert logged_model is not None, "model should be set"
    assert "a2a_agent/" in logged_model, f"model should contain 'a2a_agent/', got: {logged_model}"
    assert logged_provider == "a2a_agent", f"custom_llm_provider should be 'a2a_agent', got: {logged_provider}"

    # Verify call_type is correct for A2A
    assert call_type == "asend_message", f"call_type should be 'asend_message', got: {call_type}"

    # Verify response_cost is set to 0.0 (not None, not an error)
    # This confirms the A2A cost calculator is working
    assert response_cost is not None, "response_cost should not be None"
    assert response_cost == 0.0, f"response_cost should be 0.0 for A2A, got: {response_cost}"
