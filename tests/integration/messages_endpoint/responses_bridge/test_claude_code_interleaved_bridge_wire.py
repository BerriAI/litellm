import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue


def _expected_input(turn1: dict[str, JsonValue]) -> list[JsonValue]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": block["text"]} for block in turn1["messages"][0]["content"]],
        },
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": block["text"]} for block in turn1["messages"][1]["content"]],
        },
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}]},
        {
            "type": "function_call",
            "call_id": "toolu_a",
            "name": "Read",
            "arguments": '{"file_path": "/tmp/cc_probe/a.txt"}',
        },
        {"type": "function_call_output", "call_id": "toolu_a", "output": "ALPHA"},
        {
            "type": "message",
            "role": "system",
            "content": [
                {"type": "input_text", "text": "<total_tokens>14999970 tokens left</total_tokens>"},
                {
                    "type": "input_text",
                    "text": "First privately list what you need next; then request every item that doesn't depend on another's result in this one response.",
                },
            ],
        },
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "got A"}]},
        {
            "type": "function_call",
            "call_id": "toolu_b",
            "name": "Read",
            "arguments": '{"file_path": "/tmp/cc_probe/b.txt"}',
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "got A"}],
        },
        {"type": "function_call_output", "call_id": "toolu_b", "output": "BRAVO"},
        {
            "type": "message",
            "role": "system",
            "content": [
                {"type": "input_text", "text": "<total_tokens>14999970 tokens left</total_tokens>"},
                {
                    "type": "input_text",
                    "text": "First privately list what you need next; then request every item that doesn't depend on another's result in this one response.",
                },
            ],
        },
    ]


def test_bridge_replays_interleaved_history_in_order(gateway: Gateway) -> None:
    turn1: Final = {
        **cc.frontier_request(
            f"cache-bust-{uuid.uuid4().hex}",
            "high",
            64000,
            prompt_text="Read /tmp/cc_probe/a.txt then /tmp/cc_probe/b.txt one at a time",
        ),
        "stream": False,
    }
    turn2: Final = cc.tool_loop_turn2(
        turn1,
        (
            {"type": "thinking", "thinking": "plan", "signature": "sig_anthropic_1"},
            {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {"file_path": "/tmp/cc_probe/a.txt"}},
        ),
        (("toolu_a", "ALPHA"),),
    )
    turn3: Final = cc.tool_loop_turn2(
        turn2,
        (
            {"type": "thinking", "thinking": "got A", "signature": "sig_anthropic_2"},
            {"type": "text", "text": "got A"},
            {"type": "tool_use", "id": "toolu_b", "name": "Read", "input": {"file_path": "/tmp/cc_probe/b.txt"}},
        ),
        (("toolu_b", "BRAVO"),),
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        assert body["input"] == _expected_input(turn1), body["input"]
        return Reply(
            body=cc.responses_completed(
                "il",
                cc.OPENAI_BACKEND,
                (
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ALPHA BRAVO", "annotations": []}],
                    },
                ),
                {"input_tokens": 50, "output_tokens": 4, "total_tokens": 54},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**turn3, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
