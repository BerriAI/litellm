import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


def test_mid_loop_model_switch_replays_thinking_history_and_reasoning_betas_unchanged(gateway: Gateway) -> None:
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/hello.txt and reply with its single word",
    )
    read_call: Final = ("toolu_read_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"})
    turn2: Final = cc.tool_loop_turn2(
        turn1,
        (
            {"type": "thinking", "thinking": "need to read the file", "signature": "sig_anthropic_1"},
            {"type": "tool_use", "id": read_call[0], "name": read_call[1], "input": read_call[2]},
        ),
        ((read_call[0], "1\tPROBE\n2\t"),),
    )
    expected: Final = ({**turn1, "model": cc.FABLE}, {**turn2, "model": cc.OPUS})

    def respond(request: Request) -> Reply:
        if cc.JSON_OBJECT.validate_json(request.body) == expected[0]:
            return Reply(
                content_type="text/event-stream",
                chunks=cc.tool_use_stream(
                    f"msg_{uuid.uuid4().hex}",
                    cc.FABLE,
                    "need to read the file",
                    "sig_anthropic_1",
                    (read_call,),
                    {"input_tokens": 20, "output_tokens": 10},
                ),
            )
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(
                f"msg_{uuid.uuid4().hex}", cc.OPUS, "PROBE", {"input_tokens": 30, "output_tokens": 3}
            ),
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
        received: Final = wire.drain()
    bodies: Final = tuple(cc.JSON_OBJECT.validate_json(request.body) for request in received)
    assert bodies == expected, tuple(map(cc.body_diff, expected, bodies))
    betas: Final = tuple(cc.reasoning_betas(request.headers.get("anthropic-beta", "")) for request in received)
    assert betas == (cc.reasoning_betas(cc.FRONTIER_CLI_BETA),) * 2, betas
