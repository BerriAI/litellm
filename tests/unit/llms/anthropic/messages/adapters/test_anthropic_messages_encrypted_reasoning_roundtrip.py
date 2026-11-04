"""Encrypted reasoning round trip for `/v1/messages` served through the chat -> Responses bridge."""

import json
from typing import Final

import anthropic
import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import encrypted_reasoning_signature
from litellm.llms.anthropic.messages.adapters.handler import LiteLLMMessagesToCompletionTransformationHandler

REASONING_ITEM: Final = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque-test"}
FUNCTION_CALL_ITEM: Final = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "weather",
    "arguments": "{}",
    "status": "completed",
}
TOOL_TURN: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-5",
    "output": [REASONING_ITEM, FUNCTION_CALL_ITEM],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
}
ANSWER_TURN: Final = {
    **TOOL_TURN,
    "id": "resp_2",
    "output": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "sunny", "annotations": []}],
        }
    ],
}
REQUEST: Final = {
    "max_tokens": 256,
    "model": "openai/responses/gpt-5",
    "tools": [{"name": "weather", "description": "Weather", "input_schema": {"type": "object", "properties": {}}}],
    "thinking": {"type": "enabled", "budget_tokens": 1024},
}
USER_TURN: Final = {"role": "user", "content": "Weather?"}
TOOL_RESULT_TURN: Final = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}],
}


def _tool_turn_sse() -> bytes:
    events: Final = (
        {"type": "response.created", "response": {**TOOL_TURN, "status": "in_progress", "output": []}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**REASONING_ITEM, "encrypted_content": None},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": REASONING_ITEM},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**FUNCTION_CALL_ITEM, "arguments": "", "status": "in_progress"},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc_1", "delta": "{}"},
        {"type": "response.output_item.done", "output_index": 1, "item": FUNCTION_CALL_ITEM},
        {"type": "response.completed", "response": TOOL_TURN},
    )
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': n})}\n\n".encode()
        for n, event in enumerate(events)
    )


def _streamed_content(sse: bytes) -> list[dict[str, object]]:
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )
    client: Final = anthropic.Anthropic(api_key="transport-only", http_client=httpx.Client(transport=transport))
    with client.messages.stream(model="claude", max_tokens=16, messages=[USER_TURN]) as stream:
        return [block.model_dump(exclude_none=True) for block in stream.get_final_message().content]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_encrypted_reasoning_precedes_tool_use_and_is_replayed_once(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post("https://api.openai.com/v1/responses")
    if stream:
        upstream.respond(200, headers={"Content-Type": "text/event-stream"}, content=_tool_turn_sse())
    else:
        upstream.respond(200, json=TOOL_TURN)

    first: Final = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        messages=[USER_TURN], stream=stream, **REQUEST
    )
    content: Final = _streamed_content(b"".join([chunk async for chunk in first])) if stream else first["content"]
    assert [block["type"] for block in content if block["type"] != "text"] == ["redacted_thinking", "tool_use"]
    assert content[0]["data"] == encrypted_reasoning_signature("opaque-test")

    upstream.respond(200, json=ANSWER_TURN)
    await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        messages=[USER_TURN, {"role": "assistant", "content": content}, TOOL_RESULT_TURN], **REQUEST
    )
    replayed: Final = json.loads(upstream.calls.last.request.content)["input"]
    assert [item["type"] for item in replayed] == ["message", "reasoning", "function_call", "function_call_output"]
    assert replayed[1]["encrypted_content"] == "opaque-test"


@pytest.mark.parametrize(
    ("summary_kwargs", "bridged"),
    [
        ({}, False),
        ({"reasoningSummary": "auto"}, True),
        ({"extra_body": {"reasoningSummary": "auto"}}, True),
    ],
)
def test_bridge_detection_follows_summary_routing(summary_kwargs: dict[str, object], bridged: bool) -> None:
    completion_kwargs: Final = {
        "model": "gpt-5",
        "custom_llm_provider": "openai",
        "reasoning_effort": "high",
        **summary_kwargs,
    }
    assert LiteLLMMessagesToCompletionTransformationHandler._uses_responses_bridge(completion_kwargs) is bridged
