import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue

_INSTRUCTIONS: Final = "\n".join(block["text"] for block in cc.system_blocks())
_OUTPUT_ITEMS: Final = (
    {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "short plan"}],
        "encrypted_content": "enc_1",
    },
    {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "PONG", "annotations": []}],
    },
)


def _expected_responses_body(request_body: dict[str, JsonValue], effort: str) -> dict[str, JsonValue]:
    user_blocks: Final = request_body["messages"][0]["content"]
    expected_tools: Final = tuple(
        {
            "type": "function",
            "name": tool["name"],
            "strict": False,
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in request_body["tools"]
    )
    input_items: Final = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": block["text"]} for block in user_blocks],
        }
    ]
    for message in request_body["messages"][1:]:
        input_items.append(
            {
                "type": "message",
                "role": "system",
                "content": [
                    {"type": "input_text", "text": block["text"]}
                    for block in message["content"]
                    if block.get("type") == "text"
                ],
            }
        )
    return {
        "model": cc.OPENAI_BACKEND,
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
        "instructions": _INSTRUCTIONS,
        "max_output_tokens": request_body["max_tokens"],
        "tools": list(expected_tools),
        "reasoning": {"effort": effort},
        "stream": True,
        "user": cc.METADATA_USER_ID[:64],
        "prompt_cache_key": "00000000-0000-4000-8000-000000000000",
    }


def _assert_client_events(text: str) -> None:
    events: Final = cc.sse_events(text)
    assert [event for event, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ], [event for event, _ in events]
    assert events[1][1]["content_block"]["type"] == "thinking"
    assert events[2][1]["delta"] == {"type": "thinking_delta", "thinking": "short plan"}
    assert events[3][1]["delta"]["type"] == "signature_delta"
    assert events[6][1]["delta"] == {"type": "text_delta", "text": "PONG"}
    assert events[8][1]["delta"]["stop_reason"] == "end_turn"


def test_claude_code_frontier_body_becomes_reasoning_effort_on_responses_bridge(gateway: Gateway) -> None:
    request_body: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000)

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        assert request.headers["authorization"] == f"Bearer {cc.OPENAI_API_KEY}"
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = _expected_responses_body(request_body, "high")
        assert body == expected, {
            key: {"expected": expected.get(key), "upstream": body.get(key)}
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(
            content_type="text/event-stream", chunks=cc.responses_stream("bridge1", cc.OPENAI_BACKEND, _OUTPUT_ITEMS)
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        _assert_client_events(response.text)
        assert len(wire.drain()) == 1


def test_claude_code_legacy_thinking_budget_maps_to_reasoning_effort_on_bridge(gateway: Gateway) -> None:
    request_body: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        assert body["reasoning"] == {"effort": "high"}, body.get("reasoning")
        assert body["model"] == cc.OPENAI_BACKEND
        return Reply(
            content_type="text/event-stream", chunks=cc.responses_stream("bridge2", cc.OPENAI_BACKEND, _OUTPUT_ITEMS)
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        _assert_client_events(response.text)
        assert len(wire.drain()) == 1
