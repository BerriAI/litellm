import uuid
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue


def test_claude_code_mid_loop_model_switch_replays_history_byte_identical(gateway: Gateway) -> None:
    identity1: Final = f"msg_sw1_{uuid.uuid4().hex}"
    identity2: Final = f"msg_sw2_{uuid.uuid4().hex}"
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/hello.txt and reply with its single word",
    )
    turn2: Final = cc.tool_loop_turn2(
        turn1,
        (
            {"type": "thinking", "thinking": "need to read the file", "signature": "sig_anthropic_1"},
            {
                "type": "tool_use",
                "id": "toolu_read_1",
                "name": "Read",
                "input": {"file_path": "/tmp/cc_probe/hello.txt"},
            },
        ),
        (("toolu_read_1", "1\tPROBE\n2\t"),),
    )
    seen: list[dict[str, JsonValue]] = []

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        seen.append(body)
        if len(seen) == 1:
            assert body == {**turn1, "model": cc.FABLE}, body.get("model")
            return Reply(
                content_type="text/event-stream",
                chunks=cc.tool_use_stream(
                    identity1,
                    cc.FABLE,
                    "need to read the file",
                    "sig_anthropic_1",
                    (("toolu_read_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"}),),
                    {"input_tokens": 20, "output_tokens": 10},
                ),
            )
        expected: Final = {**turn2, "model": cc.OPUS}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity2, cc.OPUS, "PROBE", {"input_tokens": 30, "output_tokens": 3}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        fable: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        opus: Final = scenario.model(model=f"anthropic/{cc.OPUS}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response1: Final = gateway.request(
            "POST", "/v1/messages", {**turn1, "model": fable}, params={"beta": "true"}, headers=headers
        )
        assert response1.status_code == 200, response1.text
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**turn2, "model": opus}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        assert len(wire.drain()) == 2
        assert seen[1]["model"] == cc.OPUS
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT model FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity2,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["model"] == f"anthropic/{cc.OPUS}"
