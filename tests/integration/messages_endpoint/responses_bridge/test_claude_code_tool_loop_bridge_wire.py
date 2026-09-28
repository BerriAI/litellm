import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue

_INSTRUCTIONS: Final = "\n".join(block["text"] for block in cc.system_blocks())


def _turn1() -> dict[str, JsonValue]:
    return {
        **cc.frontier_request(
            f"cache-bust-{uuid.uuid4().hex}",
            "high",
            64000,
            prompt_text="Read /tmp/cc_probe/hello.txt and reply with its single word",
        ),
        "stream": False,
    }


def _upstream_items(calls: tuple[tuple[str, str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "short plan"}],
            "encrypted_content": "enc_1",
        },
        *(
            {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(tool_input),
                "status": "completed",
            }
            for i, (call_id, name, tool_input) in enumerate(calls, start=1)
        ),
    )


def _expected_turn2_input(
    turn1: dict[str, JsonValue],
    assistant_content: tuple[dict[str, JsonValue], ...],
    tool_results: tuple[tuple[str, JsonValue], ...],
) -> list[JsonValue]:
    items: Final = [
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
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "short plan"}],
            "encrypted_content": "enc_1",
        },
    ]
    items += [
        {
            "type": "function_call",
            "call_id": block["id"],
            "name": block["name"],
            "arguments": json.dumps(block["input"]),
        }
        for block in assistant_content
        if block.get("type") == "tool_use"
    ]
    items += [
        {"type": "function_call_output", "call_id": tool_use_id, "output": content}
        for tool_use_id, content in tool_results
    ]
    items.append(
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
        }
    )
    return items


def test_responses_bridge_replays_reasoning_and_tool_call_on_turn_two(gateway: Gateway) -> None:
    turn1: Final = _turn1()
    calls: Final = (("call_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"}),)
    tool_results: Final = (("call_1", "1\tPROBE\n2\t"),)
    seen: list[dict[str, JsonValue]] = []
    signature_box: list[str] = []

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        seen.append(body)
        if len(seen) == 1:
            return Reply(
                body=cc.responses_completed(
                    "tl1",
                    cc.OPENAI_BACKEND,
                    _upstream_items(calls),
                    {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
                )
            )
        assistant_content: Final = (
            {"type": "thinking", "thinking": "short plan", "signature": signature_box[0]},
            {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "/tmp/cc_probe/hello.txt"}},
        )
        assert body["input"] == _expected_turn2_input(turn1, assistant_content, tool_results), body["input"]
        return Reply(
            body=cc.responses_completed(
                "tl2",
                cc.OPENAI_BACKEND,
                (
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "PROBE", "annotations": []}],
                    },
                ),
                {"input_tokens": 50, "output_tokens": 3, "total_tokens": 53},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response1: Final = gateway.request(
            "POST", "/v1/messages", {**turn1, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response1.status_code == 200, response1.text
        payload: Final = cc.JSON_OBJECT.validate_json(response1.content)
        assert payload["stop_reason"] == "tool_use", payload
        assert payload["content"][0]["type"] == "thinking", payload["content"]
        signature: Final = payload["content"][0].get("signature")
        assert signature, payload["content"][0]
        signature_box.append(signature)
        assert payload["content"][1] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "Read",
            "input": {"file_path": "/tmp/cc_probe/hello.txt"},
        }, payload["content"]
        assistant_content: Final = (
            {"type": "thinking", "thinking": "short plan", "signature": signature},
            {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "/tmp/cc_probe/hello.txt"}},
        )
        turn2: Final = cc.tool_loop_turn2(turn1, assistant_content, tool_results)
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**turn2, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        payload2: Final = cc.JSON_OBJECT.validate_json(response2.content)
        assert payload2["stop_reason"] == "end_turn", payload2
        assert len(wire.drain()) == 2


def test_responses_bridge_replays_parallel_tool_calls_in_order(gateway: Gateway) -> None:
    turn1: Final = _turn1()
    calls: Final = (
        ("call_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"}),
        ("call_2", "Read", {"file_path": "/tmp/cc_probe/world.txt"}),
    )

    def respond(request: Request) -> Reply:
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        seen_items: Final = body["input"]
        outputs: Final = [
            item for item in seen_items if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        assert [item["call_id"] for item in outputs] == ["call_1", "call_2"], outputs
        return Reply(
            body=cc.responses_completed(
                "mt",
                cc.OPENAI_BACKEND,
                (
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "PROBE PROBE2", "annotations": []}],
                    },
                ),
                {"input_tokens": 50, "output_tokens": 4, "total_tokens": 54},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        assistant_content: Final = tuple(
            {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}
            for call_id, name, tool_input in calls
        )
        turn2: Final = cc.tool_loop_turn2(
            turn1, assistant_content, (("call_1", "1\tPROBE\n2\t"), ("call_2", "1\tPROBE2\n2\t"))
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**turn2, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
