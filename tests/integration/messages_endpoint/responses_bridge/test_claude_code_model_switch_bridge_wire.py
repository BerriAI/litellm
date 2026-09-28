import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue


def test_anthropic_signed_thinking_in_history_crosses_to_responses_bridge(gateway: Gateway) -> None:
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
    bridge_input_box: list[JsonValue] = []

    def respond_anthropic(request: Request) -> Reply:
        assert request.target == "/v1/messages", request.target
        return Reply(
            content_type="text/event-stream",
            chunks=cc.tool_use_stream(
                f"msg_sw_{uuid.uuid4().hex}",
                cc.FABLE,
                "need to read the file",
                "sig_anthropic_1",
                (("toolu_read_1", "Read", {"file_path": "/tmp/cc_probe/hello.txt"}),),
                {"input_tokens": 20, "output_tokens": 10},
            ),
        )

    def respond_openai(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        bridge_input_box.append(body.get("input"))
        reasoning_items: Final = [
            item for item in body["input"] if isinstance(item, dict) and item.get("type") == "reasoning"
        ]
        assert reasoning_items == [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "need to read the file"}]}
        ], reasoning_items
        return Reply(
            content_type="text/event-stream",
            chunks=cc.responses_stream(
                "sw",
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
            ),
        )

    with (
        wire_server(respond_anthropic) as wire_a,
        wire_server(respond_openai) as wire_b,
        gateway.scenario() as scenario,
    ):
        fable: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire_a.url, api_key=cc.ANTHROPIC_API_KEY)
        openai_alias: Final = scenario.model(
            model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire_b.url, api_key=cc.OPENAI_API_KEY
        )
        headers: Final = cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA)
        response1: Final = gateway.request(
            "POST", "/v1/messages", {**turn1, "model": fable}, params={"beta": "true"}, headers=headers
        )
        assert response1.status_code == 200, response1.text
        response2: Final = gateway.request(
            "POST", "/v1/messages", {**turn2, "model": openai_alias}, params={"beta": "true"}, headers=headers
        )
        assert response2.status_code == 200, response2.text
        events: Final = cc.sse_events(response2.text)
        assert events[-1][0] == "message_stop"
        assert len(wire_a.drain()) == 1
        assert len(wire_b.drain()) == 1
