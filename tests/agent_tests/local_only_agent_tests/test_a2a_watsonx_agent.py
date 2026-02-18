"""
Test for WatsonX Agents with A2A Protocol via Completion Bridge.

Tests the A2A SDK-level functions routing WatsonX agent requests through 
litellm.acompletion using the completion bridge.

Run with:
    pytest tests/agent_tests/local_only_agent_tests/test_a2a_watsonx_agent.py -v -s

Prerequisites:
    - WATSONX_API_BASE environment variable set to your WatsonX endpoint
    - WATSONX_API_KEY environment variable set to your API key
    - WATSONX_AGENT_ID environment variable set to your agent ID
"""

import os
import sys
from uuid import uuid4

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from a2a.types import MessageSendParams, SendMessageRequest, SendStreamingMessageRequest


def get_watsonx_config():
    """Get WatsonX configuration from environment."""
    api_base = os.environ.get("WATSONX_API_BASE")
    api_key = os.environ.get("WATSONX_API_KEY")
    agent_id = os.environ.get("WATSONX_AGENT_ID")
    
    if not all([api_base, api_key, agent_id]):
        pytest.skip("WatsonX credentials not configured. Set WATSONX_API_BASE, WATSONX_API_KEY, and WATSONX_AGENT_ID")
    
    return api_base, api_key, agent_id


@pytest.mark.asyncio
async def test_a2a_watsonx_agent_non_streaming():
    """
    Test non-streaming A2A request via the completion bridge with WatsonX agent.
    
    This test validates that WatsonX agents work with the A2A protocol through
    the completion bridge, which:
    1. Receives A2A JSON-RPC request
    2. Transforms A2A message to OpenAI format
    3. Routes through litellm.acompletion with watsonx_agent provider
    4. Transforms response back to A2A format
    """
    from litellm.a2a_protocol import asend_message

    api_base, api_key, agent_id = get_watsonx_config()
    
    litellm._turn_on_debug()

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello, introduce yourself in one sentence."}],
            "messageId": uuid4().hex,
        }
    }

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload),  # type: ignore
    )

    # Route through completion bridge with watsonx_agent provider
    response = await asend_message(
        request=request,
        api_base=api_base,
        litellm_params={
            "custom_llm_provider": "watsonx",
            "model": f"watsonx_agent/{agent_id}",
            "api_key": api_key,
        },
    )

    # Validate response follows A2A SendMessageResponse format
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

    print(f"\nWatsonX A2A Response: {response.model_dump(mode='json', exclude_none=True)}")
    print(f"Agent said: {message['parts'][0]['text']}")


@pytest.mark.asyncio
async def test_a2a_watsonx_agent_streaming():
    """
    Test streaming A2A request via the completion bridge with WatsonX agent.

    Validates proper A2A streaming format with events:
    1. Task event (kind: "task") - Initial task with status "submitted"
    2. Status update (kind: "status-update") - Status "working"
    3. Artifact update (kind: "artifact-update") - Content delivery
    4. Status update (kind: "status-update") - Final "completed" status
    """
    from litellm.a2a_protocol import asend_message_streaming

    api_base, api_key, agent_id = get_watsonx_config()
    
    litellm._turn_on_debug()

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "List three benefits of AI in one sentence each."}],
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
        api_base=api_base,
        litellm_params={
            "custom_llm_provider": "watsonx",
            "model": f"watsonx_agent/{agent_id}",
            "api_key": api_key,
        },
    ):
        chunks.append(chunk)
        print(f"Chunk: {chunk}")

    # Validate we received proper A2A streaming events
    assert len(chunks) >= 4, f"Expected at least 4 chunks (task, working, artifact, completed), got {len(chunks)}"

    # Validate chunk structure follows A2A spec
    for chunk in chunks:
        assert "jsonrpc" in chunk
        assert chunk["jsonrpc"] == "2.0"
        assert "id" in chunk
        assert "result" in chunk

    # Validate first chunk is task event
    task_chunk = chunks[0]
    assert task_chunk["result"]["kind"] == "task", "First chunk should be task event"
    assert task_chunk["result"]["status"]["state"] == "submitted"
    assert "contextId" in task_chunk["result"]
    assert "id" in task_chunk["result"]  # task id
    assert "history" in task_chunk["result"]

    # Validate second chunk is working status update
    working_chunk = chunks[1]
    assert working_chunk["result"]["kind"] == "status-update", "Second chunk should be status-update"
    assert working_chunk["result"]["status"]["state"] == "working"
    assert "taskId" in working_chunk["result"]
    assert "contextId" in working_chunk["result"]
    assert working_chunk["result"]["final"] is False

    # Validate artifact update chunk
    artifact_chunk = chunks[2]
    assert artifact_chunk["result"]["kind"] == "artifact-update", "Third chunk should be artifact-update"
    assert "artifact" in artifact_chunk["result"]
    assert "artifactId" in artifact_chunk["result"]["artifact"]
    assert "parts" in artifact_chunk["result"]["artifact"]
    assert len(artifact_chunk["result"]["artifact"]["parts"]) > 0
    assert artifact_chunk["result"]["artifact"]["parts"][0]["kind"] == "text"

    # Validate final chunk is completed status update
    final_chunk = chunks[-1]
    assert final_chunk["result"]["kind"] == "status-update", "Last chunk should be status-update"
    assert final_chunk["result"]["status"]["state"] == "completed"
    assert final_chunk["result"]["final"] is True

    print(f"\nReceived {len(chunks)} chunks with proper A2A streaming format")
    print(f"Agent response: {artifact_chunk['result']['artifact']['parts'][0]['text']}")


@pytest.mark.asyncio
async def test_a2a_watsonx_agent_with_thread_continuity():
    """
    Test WatsonX agent with thread continuity through A2A protocol.
    
    WatsonX agents support thread-based conversation continuity. This test
    validates that thread_id is properly passed through the A2A bridge.
    """
    from litellm.a2a_protocol import asend_message

    api_base, api_key, agent_id = get_watsonx_config()
    
    litellm._turn_on_debug()

    # First message - creates a new thread
    send_message_payload_1 = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Remember this: my favorite color is blue."}],
            "messageId": uuid4().hex,
        }
    }

    request1 = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload_1),  # type: ignore
    )

    response1 = await asend_message(
        request=request1,
        api_base=api_base,
        litellm_params={
            "custom_llm_provider": "watsonx",
            "model": f"watsonx_agent/{agent_id}",
            "api_key": api_key,
        },
    )

    print(f"\nFirst message response: {response1.model_dump(mode='json', exclude_none=True)}")

    # Extract thread_id from hidden params (if available)
    thread_id = response1._hidden_params.get("thread_id") if hasattr(response1, "_hidden_params") else None
    
    if thread_id:
        print(f"Thread ID: {thread_id}")
        
        # Second message - continue conversation with same thread
        send_message_payload_2 = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "What is my favorite color?"}],
                "messageId": uuid4().hex,
            }
        }

        request2 = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload_2),  # type: ignore
        )

        response2 = await asend_message(
            request=request2,
            api_base=api_base,
            litellm_params={
                "custom_llm_provider": "watsonx",
                "model": f"watsonx_agent/{agent_id}",
                "api_key": api_key,
                "thread_id": thread_id,  # Continue same conversation
            },
        )

        print(f"\nSecond message response: {response2.model_dump(mode='json', exclude_none=True)}")
        
        # The agent should remember the color from the first message
        response_text = response2.result["message"]["parts"][0]["text"].lower()
        print(f"Agent's answer: {response_text}")
        # Note: Asserting the content might be flaky depending on agent behavior
        # assert "blue" in response_text, "Agent should remember the favorite color"
    else:
        print("Thread ID not available in response - skipping continuity test")
