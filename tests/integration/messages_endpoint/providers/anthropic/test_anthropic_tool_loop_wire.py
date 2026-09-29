import uuid
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration._support import claude_code as cc
from pydantic import JsonValue

_THINKING: Final = "need to read the file"
_SIGNATURE: Final = "sig_probe_1"
_USAGE: Final = {"input_tokens": 20, "output_tokens": 10}


def _diff(expected: dict[str, JsonValue], body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: {"expected": expected.get(key), "upstream": body.get(key)}
        for key in expected.keys() | body.keys()
        if expected.get(key) != body.get(key)
    }


def test_tool_loop_round_trips_thinking_tool_use_and_tool_result(gateway: Gateway) -> None:
    identity1: Final = f"msg_tl1_{uuid.uuid4().hex}"
    identity2: Final = f"msg_tl2_{uuid.uuid4().hex}"
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/hello.txt and reply with its single word",
    )
    calls: Final = (("toolu_read_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"}),)
    turn2: Final = cc.tool_loop_turn2(
        turn1,
        (
            {"type": "thinking", "thinking": _THINKING, "signature": _SIGNATURE},
            {
                "type": "tool_use",
                "id": "toolu_read_1",
                "name": "Read",
                "input": {"file_path": "/tmp/cc_probe/hello.txt"},
            },
        ),
        (("toolu_read_1", "1\tPROBE\n2\t"),),
    )
    first_expected: Final = {**turn1, "model": cc.FABLE}
    second_expected: Final = {**turn2, "model": cc.FABLE}

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        if body == first_expected:
            return Reply(
                content_type="text/event-stream",
                chunks=cc.tool_use_stream(identity1, cc.FABLE, _THINKING, _SIGNATURE, calls, _USAGE),
            )
        assert body == second_expected, _diff(second_expected, body)
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity2, cc.FABLE, "PROBE", {"input_tokens": 30, "output_tokens": 3}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response1: Final = gateway.request(
            "POST", "/v1/messages", {**turn1, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response1.status_code == 200, response1.text
        events: Final = cc.sse_events(response1.text)
        assert [
            (
                event,
                data.get("delta", {}).get(
                    "type", data.get("content_block", {}).get("type", data.get("delta", {}).get("stop_reason"))
                ),
            )
            for event, data in events
        ] == [
            ("message_start", None),
            ("content_block_start", "thinking"),
            ("content_block_delta", "thinking_delta"),
            ("content_block_delta", "signature_delta"),
            ("content_block_stop", None),
            ("content_block_start", "tool_use"),
            ("content_block_delta", "input_json_delta"),
            ("content_block_delta", "input_json_delta"),
            ("content_block_stop", None),
            ("message_delta", "tool_use"),
            ("message_stop", None),
        ]
        assert events[5][1]["content_block"]["id"] == "toolu_read_1"
        assert events[5][1]["content_block"]["name"] == "Read"
        partial: Final = events[6][1]["delta"]["partial_json"] + events[7][1]["delta"]["partial_json"]
        assert partial == '{"file_path": "/tmp/cc_probe/hello.txt"}'
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**turn2, "model": model}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        events2: Final = cc.sse_events(response2.text)
        assert events2[2][1]["delta"] == {"type": "text_delta", "text": "PROBE"}
        assert events2[4][1]["delta"]["stop_reason"] == "end_turn"
        bodies: Final = tuple(cc.JSON_OBJECT.validate_json(request.body) for request in wire.drain())
        assert bodies == (first_expected, second_expected), bodies
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity2,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == 30


def test_parallel_tool_results_reach_anthropic_in_client_order(gateway: Gateway) -> None:
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/hello.txt and /tmp/cc_probe/world.txt and reply with both words",
    )
    turn2: Final = cc.tool_loop_turn2(
        turn1,
        (
            {"type": "thinking", "thinking": _THINKING, "signature": _SIGNATURE},
            {
                "type": "tool_use",
                "id": "toolu_read_1",
                "name": "Read",
                "input": {"file_path": "/tmp/cc_probe/hello.txt"},
            },
            {
                "type": "tool_use",
                "id": "toolu_read_2",
                "name": "Read",
                "input": {"file_path": "/tmp/cc_probe/world.txt"},
            },
        ),
        (("toolu_read_2", "1\tPROBE2\n2\t"), ("toolu_read_1", "1\tPROBE\n2\t")),
    )

    def respond(request: Request) -> Reply:
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**turn2, "model": cc.FABLE}
        assert body == expected, _diff(expected, body)
        results: Final = [block for block in body["messages"][3]["content"] if block["type"] == "tool_result"]
        assert [block["tool_use_id"] for block in results] == ["toolu_read_2", "toolu_read_1"]
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(f"msg_mt_{uuid.uuid4().hex}", cc.FABLE, "PROBE PROBE2", _USAGE),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**turn2, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
