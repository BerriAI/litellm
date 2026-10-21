import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


def test_mid_loop_model_switch_replays_thinking_history_and_reasoning_unchanged(gateway: Gateway) -> None:
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

    def respond(request: Request) -> Reply:
        if b"tool_result" not in request.body:
            return Reply(
                content_type="text/event-stream",
                chunks=cc.tool_use_stream(
                    f"msg_{uuid.uuid4().hex}",
                    cc.FABLE,
                    "need to read the file",
                    "sig_anthropic_1",
                    (("toolu_read_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"}),),
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
    assert len(received) == 2, received
    to_fable: Final = cc.forwarded(turn1, received[0])
    to_opus: Final = cc.forwarded(turn2, received[1])
    assert (to_fable.model, to_opus.model) == (cc.FABLE, cc.OPUS), received
    assert to_opus.assistant_history == (
        [
            {"type": "thinking", "thinking": "need to read the file", "signature": "sig_anthropic_1"},
            {
                "type": "tool_use",
                "id": "toolu_read_1",
                "name": "Read",
                "input": {"file_path": "/tmp/cc_probe/hello.txt"},
            },
        ],
    ), to_opus.assistant_history
    adaptive_high: Final = {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}}
    assert (to_fable.reasoning, to_opus.reasoning) == (adaptive_high, adaptive_high)
    assert (to_fable.other_changes, to_opus.other_changes) == ({}, {})
    assert (to_fable.reasoning_betas, to_opus.reasoning_betas) == (cc.CLAUDE_CODE_REASONING_BETAS,) * 2
