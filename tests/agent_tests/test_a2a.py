"""
Test for LiteLLM A2A module.

Run with:
    pytest tests/test_litellm/a2a/test_a2a.py -v -s
"""

from uuid import uuid4
import litellm
import pytest
import os
import sys
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
    from litellm.a2a import asend_message, create_a2a_client

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
