"""
Simple A2A agent tests - non-streaming and streaming.

These tests validate the localhost URL retry logic: if an A2A agent's card
contains a localhost/internal URL (e.g., http://0.0.0.0:8001/), the request
will fail with a connection error. LiteLLM detects this and automatically
retries using the original api_base URL instead.

Requires A2A_AGENT_URL environment variable to be set.

Run with:
    A2A_AGENT_URL=https://your-agent.example.com pytest tests/agent_tests/test_a2a_agent.py -v -s
"""

import os

import pytest
from uuid import uuid4


def get_a2a_agent_url():
    """Get A2A agent URL from environment, skip test if not set."""
    url = os.environ.get("A2A_AGENT_URL")
    return url


@pytest.mark.asyncio
async def test_a2a_non_streaming():
    """Test non-streaming A2A request."""
    from a2a.types import MessageSendParams, SendMessageRequest
    from litellm.a2a_protocol import asend_message

    api_base = get_a2a_agent_url()

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Say hello in one word"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    response = await asend_message(
        request=request,
        api_base=api_base,
    )

    assert response is not None
    print(f"\nNon-streaming response: {response}")


@pytest.mark.asyncio
async def test_a2a_streaming():
    """Test streaming A2A request."""
    from a2a.types import MessageSendParams, SendStreamingMessageRequest
    from litellm.a2a_protocol import asend_message_streaming

    api_base = get_a2a_agent_url()

    request = SendStreamingMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Say hello in one word"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    chunks = []
    async for chunk in asend_message_streaming(
        request=request,
        api_base=api_base,
    ):
        chunks.append(chunk)
        print(f"\nStreaming chunk: {chunk}")

    assert len(chunks) > 0, "Should receive at least one chunk"
    print(f"\nTotal chunks received: {len(chunks)}")
