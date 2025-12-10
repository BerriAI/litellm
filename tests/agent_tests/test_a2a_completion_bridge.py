"""
Test for A2A to LiteLLM Completion Bridge.

Tests the SDK-level functions that route A2A requests through litellm.acompletion.

Run with:
    pytest tests/agent_tests/test_a2a_completion_bridge.py -v -s

Prerequisites:
    - LangGraph server running on localhost:2024
"""

import os
import sys
from uuid import uuid4

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from a2a.types import MessageSendParams, SendMessageRequest, SendStreamingMessageRequest


@pytest.mark.asyncio
async def test_a2a_completion_bridge_non_streaming():
    """
    Test non-streaming A2A request via the completion bridge with LangGraph provider.
    """
    from litellm.a2a_protocol import asend_message

    litellm._turn_on_debug()

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "What is 2 + 2?"}],
            "messageId": uuid4().hex,
        }
    }

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload),  # type: ignore
    )

    response = await asend_message(
        request=request,
        api_base="http://localhost:2024",
        litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
    )

    # Validate response is LiteLLMSendMessageResponse
    assert response.jsonrpc == "2.0"
    assert response.id is not None
    assert response.result is not None
    assert "message" in response.result

    message = response.result["message"]
    assert "role" in message
    assert message["role"] == "agent"
    assert "parts" in message
    assert len(message["parts"]) > 0
    assert message["parts"][0]["kind"] == "text"
    assert len(message["parts"][0]["text"]) > 0

    print(f"Response: {response.model_dump(mode='json', exclude_none=True)}")


@pytest.mark.asyncio
async def test_a2a_completion_bridge_streaming():
    """
    Test streaming A2A request via the completion bridge with LangGraph provider.
    """
    from litellm.a2a_protocol import asend_message_streaming

    litellm._turn_on_debug()

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Count from 1 to 5."}],
            "messageId": uuid4().hex,
        }
    }

    request = SendStreamingMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload),  # type: ignore
    )

    chunks = []
    async for chunk in asend_message_streaming(
        request=request,
        api_base="http://localhost:2024",
        litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
    ):
        chunks.append(chunk)
        print(f"Chunk: {chunk}")

    # Validate we received chunks
    assert len(chunks) > 0

    # Validate chunk structure (chunks are dicts from bridge)
    for chunk in chunks:
        assert "jsonrpc" in chunk
        assert chunk["jsonrpc"] == "2.0"
        assert "result" in chunk
        assert "message" in chunk["result"]
        message = chunk["result"]["message"]
        assert "role" in message
        assert message["role"] == "agent"
        assert "parts" in message

    print(f"Received {len(chunks)} chunks")

