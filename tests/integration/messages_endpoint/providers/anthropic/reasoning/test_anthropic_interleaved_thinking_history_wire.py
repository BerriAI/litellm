import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_TURN1_TOOL: Final = ("toolu_a", "Read", {"file_path": "/tmp/cc_probe/a.txt"})
_TURN2_TOOL: Final = ("toolu_b", "Read", {"file_path": "/tmp/cc_probe/b.txt"})


def _diff(expected: dict[str, JsonValue], body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: {"expected": expected.get(key), "upstream": body.get(key)}
        for key in expected.keys() | body.keys()
        if expected.get(key) != body.get(key)
    }


def _turn2(base: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cc.tool_loop_turn2(
        base,
        (
            {"type": "thinking", "thinking": "plan", "signature": "sig1"},
            {"type": "tool_use", "id": _TURN1_TOOL[0], "name": _TURN1_TOOL[1], "input": _TURN1_TOOL[2]},
        ),
        ((_TURN1_TOOL[0], "ALPHA"),),
    )


def _turn3(turn2: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cc.tool_loop_turn2(
        turn2,
        (
            {"type": "thinking", "thinking": "got A", "signature": "sig2"},
            {"type": "text", "text": "got A"},
            {"type": "tool_use", "id": _TURN2_TOOL[0], "name": _TURN2_TOOL[1], "input": _TURN2_TOOL[2]},
        ),
        ((_TURN2_TOOL[0], "BRAVO"),),
    )


def _interleaved_stream(identity: str) -> tuple[bytes, ...]:
    usage: Final = {"input_tokens": 20, "output_tokens": 12}
    return (
        cc.sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": cc.FABLE,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 1},
                },
            },
        ),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "got A"}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig2"}},
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "got A"}},
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        cc.sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": _TURN2_TOOL[0], "name": _TURN2_TOOL[1], "input": {}},
            },
        ),
        cc.sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "/tmp/cc_pr'},
            },
        ),
        cc.sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": 'obe/b.txt"}'},
            },
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 2}),
        cc.sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        cc.sse_frame("message_stop", {"type": "message_stop"}),
    )


def test_interleaved_thinking_text_and_tool_use_history_reaches_anthropic_identical(gateway: Gateway) -> None:
    identity: Final = f"msg_il_{uuid.uuid4().hex}"
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/a.txt then /tmp/cc_probe/b.txt one at a time and reply with both words",
    )
    turn2: Final = _turn2(turn1)
    turn3: Final = _turn3(turn2)
    turn2_expected: Final = {**turn2, "model": cc.FABLE}
    turn3_expected: Final = {**turn3, "model": cc.FABLE}

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        upstream_beta: Final = request.headers.get("anthropic-beta", "")
        assert upstream_beta.split(",").count("interleaved-thinking-2025-05-14") == 1, upstream_beta
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        if body == turn2_expected:
            return Reply(
                content_type="text/event-stream",
                chunks=cc.text_stream("msg_il_turn2", cc.FABLE, "got A", {"input_tokens": 20, "output_tokens": 4}),
            )
        assert body == turn3_expected, _diff(turn3_expected, body)
        return Reply(content_type="text/event-stream", chunks=_interleaved_stream(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**turn2, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        response3: Final = gateway.request(
            "POST", "/v1/messages", {**turn3, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response3.status_code == 200, response3.text
        events: Final = cc.sse_events(response3.text)
        started: Final = [
            (data["index"], data["content_block"]["type"]) for event, data in events if event == "content_block_start"
        ]
        assert started == [(0, "thinking"), (1, "text"), (2, "tool_use")], started
        assert events[-1][0] == "message_stop"
        bodies: Final = tuple(cc.JSON_OBJECT.validate_json(request.body) for request in wire.drain())
        assert bodies == (turn2_expected, turn3_expected), bodies
