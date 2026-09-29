import uuid
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_USAGE: Final = {
    "input_tokens": 10,
    "cache_read_input_tokens": 3000,
    "cache_creation_input_tokens": 200,
    "output_tokens": 5,
}


def test_cached_turn_charges_cache_read_and_creation_rates(gateway: Gateway) -> None:
    identity: Final = f"msg_pc_{uuid.uuid4().hex}"
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
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**turn2, "model": cc.FABLE}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(content_type="text/event-stream", chunks=cc.text_stream(identity, cc.FABLE, "PROBE", _USAGE))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{cc.FABLE}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            input_cost_per_token=1e-6,
            output_cost_per_token=5e-6,
            cache_read_input_token_cost=1e-7,
            cache_creation_input_token_cost=1.25e-6,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**turn2, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        events: Final = cc.sse_events(response.text)
        usage: Final = events[0][1]["message"]["usage"]
        assert usage["cache_read_input_tokens"] == 3000, usage
        assert usage["cache_creation_input_tokens"] == 200, usage
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(10 * 1e-6 + 3000 * 1e-7 + 200 * 1.25e-6 + 5 * 5e-6), dict(
            rows[0]
        )
