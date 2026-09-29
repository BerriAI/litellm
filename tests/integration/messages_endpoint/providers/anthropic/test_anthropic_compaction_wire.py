import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration._support import claude_code as cc
from pydantic import JsonValue

_CONTEXT_MANAGEMENT: Final = {
    "edits": [
        {"type": "clear_thinking_20251015", "keep": "all"},
        {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}},
    ]
}
_COMPACTION_BLOCK: Final = {"type": "compaction", "content": "<summary>"}


def _compaction_stream(identity: str) -> tuple[bytes, ...]:
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
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": dict(_COMPACTION_BLOCK)},
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        cc.sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
                "context_management": {
                    "applied_edits": [{"type": "compact_20260112", "compacted_at": "2026-09-26T00:00:00Z"}]
                },
            },
        ),
        cc.sse_frame("message_stop", {"type": "message_stop"}),
    )


def test_compaction_edit_and_applied_edit_block_round_trip_through_anthropic(gateway: Gateway) -> None:
    identity: Final = f"msg_cm_{uuid.uuid4().hex}"
    request_body: Final = {
        **cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000),
        "context_management": _CONTEXT_MANAGEMENT,
    }
    turn3: Final = cc.tool_loop_turn2(
        request_body,
        (
            dict(_COMPACTION_BLOCK),
            {"type": "text", "text": "continuing after compaction"},
        ),
        (),
    )
    first_expected: Final = {**request_body, "model": cc.FABLE}
    second_expected: Final = {**turn3, "model": cc.FABLE}

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        if body == first_expected:
            assert body["context_management"] == _CONTEXT_MANAGEMENT
            return Reply(content_type="text/event-stream", chunks=_compaction_stream(identity))
        assert body == second_expected, {
            key: (second_expected.get(key), body.get(key))
            for key in second_expected.keys() | body.keys()
            if second_expected.get(key) != body.get(key)
        }
        assert body["context_management"] == _CONTEXT_MANAGEMENT
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream("msg_cm_next", cc.FABLE, "OK", {"input_tokens": 20, "output_tokens": 2}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response1: Final = gateway.request(
            "POST", "/v1/messages", {**request_body, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response1.status_code == 200, response1.text
        events: Final = cc.sse_events(response1.text)
        assert events[1][1]["content_block"] == _COMPACTION_BLOCK, events[1]
        deltas: Final = [data for event, data in events if event == "message_delta"]
        assert len(deltas) == 1 and deltas[0].get("context_management") == {
            "applied_edits": [{"type": "compact_20260112", "compacted_at": "2026-09-26T00:00:00Z"}]
        }, events
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**turn3, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        bodies: Final = tuple(cc.JSON_OBJECT.validate_json(request.body) for request in wire.drain())
        assert bodies == (first_expected, second_expected), bodies
