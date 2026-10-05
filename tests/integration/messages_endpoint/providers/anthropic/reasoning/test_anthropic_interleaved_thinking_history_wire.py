import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_READ_A: Final = {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {"file_path": "/tmp/cc_probe/a.txt"}}
_READ_B: Final = {"type": "tool_use", "id": "toolu_b", "name": "Read", "input": {"file_path": "/tmp/cc_probe/b.txt"}}


def test_interleaved_thinking_history_reaches_anthropic_and_interleaved_blocks_stream_back(gateway: Gateway) -> None:
    turn1: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Read /tmp/cc_probe/a.txt then /tmp/cc_probe/b.txt one at a time and reply with both words",
    )
    turn2: Final = cc.tool_loop_turn2(
        turn1, ({"type": "thinking", "thinking": "plan", "signature": "sig1"}, _READ_A), (("toolu_a", "ALPHA"),)
    )
    turn3: Final = cc.tool_loop_turn2(
        turn2,
        ({"type": "thinking", "thinking": "got A", "signature": "sig2"}, {"type": "text", "text": "got A"}, _READ_B),
        (("toolu_b", "BRAVO"),),
    )

    def respond(request: Request) -> Reply:
        if b"toolu_b" not in request.body:
            return Reply(
                content_type="text/event-stream",
                chunks=cc.text_stream("msg_il_turn2", cc.FABLE, "got A", {"input_tokens": 20, "output_tokens": 4}),
            )
        return Reply(
            content_type="text/event-stream",
            chunks=cc.message_stream(
                f"msg_il_{uuid.uuid4().hex}",
                cc.FABLE,
                (
                    {"type": "thinking", "thinking": "got B", "signature": "sig3"},
                    {"type": "text", "text": "got B"},
                    {
                        "type": "tool_use",
                        "id": "toolu_c",
                        "name": "Read",
                        "input": {"file_path": "/tmp/cc_probe/c.txt"},
                    },
                ),
                {"input_tokens": 20, "output_tokens": 12},
            ),
        )

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
        received: Final = wire.drain()
    assert len(received) == 2, received
    second: Final = cc.forwarded(turn2, received[0])
    third: Final = cc.forwarded(turn3, received[1])
    assert second.assistant_history == ([{"type": "thinking", "thinking": "plan", "signature": "sig1"}, _READ_A],)
    assert third.assistant_history == (
        [{"type": "thinking", "thinking": "plan", "signature": "sig1"}, _READ_A],
        [{"type": "thinking", "thinking": "got A", "signature": "sig2"}, {"type": "text", "text": "got A"}, _READ_B],
    ), third.assistant_history
    adaptive_high: Final = {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}}
    assert (second.reasoning, third.reasoning) == (adaptive_high, adaptive_high)
    assert (second.other_changes, third.other_changes) == ({}, {})
    assert (second.reasoning_betas, third.reasoning_betas) == (cc.CLAUDE_CODE_REASONING_BETAS,) * 2
    assert cc.streamed_content(response3.text) == [
        {"type": "thinking", "thinking": "got B", "signature": "sig3"},
        {"type": "text", "text": "got B"},
        {"type": "tool_use", "id": "toolu_c", "name": "Read", "input": {"file_path": "/tmp/cc_probe/c.txt"}},
    ], response3.text
