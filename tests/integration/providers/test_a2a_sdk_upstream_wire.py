"""A2A SDK paths against a scripted upstream agent.

Ports the legacy agent_tests coverage: asend_message with an explicit A2A
client, the LiteLLM logging payload for A2A calls, the pydantic_ai_agents
provider's send-and-poll flow, and the completion bridge driving LangGraph
/runs/wait and /runs/stream endpoints.
"""

import asyncio
import json
from typing import Final
from uuid import uuid4

import pytest
from a2a.compat.v0_3.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

import litellm
from tests.integration._support.wire import Reply, Request, wire_server


def _agent_card(wire_url: str) -> dict:
    return {
        "protocolVersion": "0.3",
        "name": "scripted-a2a-agent",
        "description": "scripted upstream",
        "version": "1.0.0",
        "url": wire_url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _a2a_message_response(request: Request, text: str) -> Reply:
    body = json.loads(request.body)
    return Reply(
        body=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "kind": "message",
                    "role": "agent",
                    "messageId": uuid4().hex,
                    "parts": [{"kind": "text", "text": text}],
                },
            }
        ).encode()
    )


def _send_request() -> SendMessageRequest:
    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello from the test"}],
            "messageId": uuid4().hex,
        },
    }
    return SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))


@pytest.mark.asyncio
async def test_asend_message_with_client_decorator():
    """asend_message over an explicit A2A client returns the agent message."""
    from litellm.a2a_protocol import asend_message, create_a2a_client

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target in ("/.well-known/agent-card.json", "/.well-known/agent.json")
            return Reply(body=json.dumps(_agent_card(wire.url)).encode())
        return _a2a_message_response(request, "scripted agent reply")

    with wire_server(upstream) as wire:
        a2a_client = await create_a2a_client(base_url=wire.url)
        response = await asend_message(a2a_client=a2a_client, request=_send_request())

    assert response is not None
    result = response.result if hasattr(response, "result") else response
    result_dict = result if isinstance(result, dict) else result.model_dump(mode="python", exclude_none=True)
    assert result_dict["kind"] == "message"
    assert result_dict["role"] == "agent"
    assert result_dict["parts"][0]["text"] == "scripted agent reply"


@pytest.mark.asyncio
async def test_a2a_logging_payload():
    """A2A calls produce a standard logging payload with provider and zero cost."""
    from litellm.integrations.custom_logger import CustomLogger

    class TestA2ALogger(CustomLogger):
        def __init__(self):
            self.standard_logging_payload = None
            self.log_success_called = False
            super().__init__()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.log_success_called = True
            self.standard_logging_payload = kwargs.get("standard_logging_object", None)

    litellm.logging_callback_manager._reset_all_callbacks()
    test_logger = TestA2ALogger()
    litellm.callbacks = [test_logger]

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_agent_card(wire.url)).encode())
        return _a2a_message_response(request, "scripted agent reply")

    try:
        from litellm.a2a_protocol import asend_message, create_a2a_client

        with wire_server(upstream) as wire:
            a2a_client = await create_a2a_client(base_url=wire.url)
            await asend_message(a2a_client=a2a_client, request=_send_request())

        deadline = asyncio.get_event_loop().time() + 10
        while not test_logger.log_success_called and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
    finally:
        litellm.logging_callback_manager._reset_all_callbacks()

    assert test_logger.log_success_called is True
    slp = test_logger.standard_logging_payload
    assert slp is not None

    logged_model = slp.get("model") if isinstance(slp, dict) else getattr(slp, "model", None)
    logged_provider = (
        slp.get("custom_llm_provider") if isinstance(slp, dict) else getattr(slp, "custom_llm_provider", None)
    )
    call_type = slp.get("call_type") if isinstance(slp, dict) else getattr(slp, "call_type", None)
    response_cost = (
        slp.get("response_cost") if isinstance(slp, dict) else getattr(slp, "response_cost", None)
    )

    assert logged_model is not None and "a2a_agent/" in logged_model
    assert logged_provider == "a2a_agent"
    assert call_type == "asend_message"
    assert response_cost == 0.0


def _pydantic_task_response(request: Request) -> Reply:
    body = json.loads(request.body)
    assert body["method"] in ("message/send", "tasks/get"), body["method"]
    return Reply(
        body=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "id": "task-1",
                    "kind": "task",
                    "status": {"state": "completed"},
                    "history": [
                        body["params"]["message"] if body["method"] == "message/send" else {},
                        {
                            "role": "agent",
                            "kind": "message",
                            "messageId": "agent-msg-1",
                            "parts": [{"kind": "text", "text": "pydantic agent reply"}],
                        },
                    ],
                    "artifacts": [
                        {
                            "artifactId": "artifact-1",
                            "parts": [{"kind": "text", "text": "pydantic agent reply"}],
                        }
                    ],
                },
            }
        ).encode()
    )


@pytest.mark.asyncio
async def test_pydantic_ai_non_streaming():
    """pydantic_ai_agents send returns the completed task with agent text."""
    from litellm.a2a_protocol import asend_message

    with wire_server(_pydantic_task_response) as wire:
        response = await asend_message(
            request=_send_request(),
            api_base=wire.url,
            litellm_params={"custom_llm_provider": "pydantic_ai_agents"},
        )

    assert response is not None
    result = response.result
    assert result is not None
    result_dict = result if isinstance(result, dict) else result.model_dump(mode="python", exclude_none=True)
    assert result_dict["kind"] == "message"
    assert result_dict["role"] == "agent"
    assert result_dict["parts"][0]["kind"] == "text"
    assert result_dict["parts"][0]["text"] == "pydantic agent reply"


@pytest.mark.asyncio
async def test_pydantic_ai_fake_streaming():
    """pydantic_ai_agents fake streaming emits task, working, artifact, completed."""
    from litellm.a2a_protocol import asend_message_streaming

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello from Pydantic AI streaming test!"}],
            "messageId": uuid4().hex,
        },
    }
    request = SendStreamingMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))

    with wire_server(_pydantic_task_response) as wire:
        chunks = []
        async for chunk in asend_message_streaming(
            request=request,
            api_base=wire.url,
            litellm_params={"custom_llm_provider": "pydantic_ai_agents"},
        ):
            chunk_dict = chunk.model_dump(mode="json", exclude_none=True) if hasattr(chunk, "model_dump") else chunk
            chunks.append(chunk_dict)

    kinds = [c.get("result", {}).get("kind") for c in chunks]
    states = [c.get("result", {}).get("status", {}).get("state") for c in chunks]
    assert len(chunks) > 0
    assert "task" in kinds
    assert "artifact-update" in kinds
    assert "working" in states
    assert "completed" in states


def _langgraph_upstream(request: Request) -> Reply:
    assert request.method == "POST"
    body = json.loads(request.body)
    messages = body["input"]["messages"]
    assert messages[0]["role"] == "human"
    if request.target == "/runs/wait":
        return Reply(
            body=json.dumps(
                {"messages": [{"type": "ai", "role": "assistant", "content": "2 + 2 = 4."}]}
            ).encode()
        )
    assert request.target == "/runs/stream"
    sse = (
        'data: {"content":"1, "}\n\n'
        'data: {"content":"2, 3, 4, 5."}\n\n'
    )
    return Reply(body=sse.encode(), content_type="text/event-stream")


@pytest.mark.asyncio
async def test_a2a_completion_bridge_non_streaming():
    """Completion bridge non-streaming returns the LangGraph text as an A2A message."""
    from litellm.a2a_protocol import asend_message

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "What is 2 + 2?"}],
            "messageId": uuid4().hex,
        },
    }
    request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))

    with wire_server(_langgraph_upstream) as wire:
        response = await asend_message(
            request=request,
            api_base=wire.url,
            litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
        )

    assert response.jsonrpc == "2.0"
    assert response.id is not None
    assert response.result is not None
    assert response.result.get("kind") == "message"

    message = response.result
    assert message["role"] == "agent"
    assert len(message["parts"]) > 0
    assert message["parts"][0]["kind"] == "text"
    assert message["parts"][0]["text"] == "2 + 2 = 4."


@pytest.mark.asyncio
async def test_a2a_completion_bridge_streaming():
    """Completion bridge streaming emits task, working, artifact, completed events."""
    from litellm.a2a_protocol import asend_message_streaming

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Count from 1 to 5."}],
            "messageId": uuid4().hex,
        },
    }
    request = SendStreamingMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))

    with wire_server(_langgraph_upstream) as wire:
        chunks = []
        async for chunk in asend_message_streaming(
            request=request,
            api_base=wire.url,
            litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
        ):
            chunks.append(chunk)

    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    for chunk in chunks:
        assert chunk["jsonrpc"] == "2.0"
        assert "id" in chunk
        assert "result" in chunk

    task_chunk = chunks[0]
    assert task_chunk["result"]["kind"] == "task"
    assert task_chunk["result"]["status"]["state"] == "submitted"

    working_chunk = chunks[1]
    assert working_chunk["result"]["kind"] == "status-update"
    assert working_chunk["result"]["status"]["state"] == "working"
    assert working_chunk["result"]["final"] is False

    artifact_chunk = chunks[2]
    assert artifact_chunk["result"]["kind"] == "artifact-update"
    parts = artifact_chunk["result"]["artifact"]["parts"]
    assert len(parts) > 0
    assert parts[0]["kind"] == "text"

    final_chunk = chunks[-1]
    assert final_chunk["result"]["kind"] == "status-update"
    assert final_chunk["result"]["status"]["state"] == "completed"
    assert final_chunk["result"]["final"] is True
