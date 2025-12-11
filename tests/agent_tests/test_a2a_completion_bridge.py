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

    Validates proper A2A streaming format with events:
    1. Task event (kind: "task") - Initial task with status "submitted"
    2. Status update (kind: "status-update") - Status "working"
    3. Artifact update (kind: "artifact-update") - Content delivery
    4. Status update (kind: "status-update") - Final "completed" status
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

    print(f"Received {len(chunks)} chunks with proper A2A streaming format")


@pytest.mark.asyncio
async def test_a2a_completion_bridge_bedrock_agentcore():
    """
    Test A2A request via the completion bridge with Bedrock AgentCore provider.
    
    Uses the AgentCore runtime ARN to call a hosted agent.
    """
    from litellm.a2a_protocol import asend_message_streaming

    litellm._turn_on_debug()

    # Bedrock AgentCore ARN (streaming-capable runtime)
    agentcore_arn = "arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC"

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Explain machine learning in simple terms"}],
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
        api_base=None,  # Not needed for Bedrock AgentCore
        litellm_params={
            "custom_llm_provider": "bedrock",
            "model": f"bedrock/agentcore/{agentcore_arn}",
        },
    ):
        chunks.append(chunk)
        print(f"Chunk: {chunk}")

    # Validate we received proper A2A streaming events
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    # Validate first chunk is task event
    assert chunks[0]["result"]["kind"] == "task"
    assert chunks[0]["result"]["status"]["state"] == "submitted"

    # Validate final chunk is completed status
    assert chunks[-1]["result"]["kind"] == "status-update"
    assert chunks[-1]["result"]["status"]["state"] == "completed"
    assert chunks[-1]["result"]["final"] is True

    print(f"Received {len(chunks)} chunks from Bedrock AgentCore")

