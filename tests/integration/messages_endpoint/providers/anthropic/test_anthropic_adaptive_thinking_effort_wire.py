import uuid
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


def _diff(expected: dict[str, JsonValue], body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: {"expected": expected.get(key), "upstream": body.get(key)}
        for key in expected.keys() | body.keys()
        if expected.get(key) != body.get(key)
    }


def test_adaptive_thinking_and_effort_reach_anthropic_intact(gateway: Gateway) -> None:
    identity: Final = f"msg_fable_{uuid.uuid4().hex}"
    request_body: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000)
    cli_beta: Final = frozenset(cc.FRONTIER_CLI_BETA.split(","))

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == cc.ANTHROPIC_API_KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        upstream_beta: Final = request.headers.get("anthropic-beta", "")
        assert cli_beta <= frozenset(upstream_beta.split(",")), upstream_beta
        assert upstream_beta.split(",").count("effort-2025-11-24") == 1, upstream_beta
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**request_body, "model": cc.FABLE}
        assert body == expected, _diff(expected, body)
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity, cc.FABLE, "PONG", {"input_tokens": 12, "output_tokens": 4}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        events: Final = cc.sse_events(response.text)
        assert [event for event, _ in events][-1] == "message_stop"
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == 12 and rows[0]["completion_tokens"] == 4


def test_xhigh_effort_reaches_anthropic_and_charges_by_usage(gateway: Gateway) -> None:
    identity: Final = f"msg_opus_{uuid.uuid4().hex}"
    request_body: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "xhigh", 128000)
    cli_beta: Final = frozenset(cc.FRONTIER_CLI_BETA.split(","))

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == cc.ANTHROPIC_API_KEY
        upstream_beta: Final = request.headers.get("anthropic-beta", "")
        assert cli_beta <= frozenset(upstream_beta.split(",")), upstream_beta
        assert upstream_beta.split(",").count("effort-2025-11-24") == 1, upstream_beta
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**request_body, "model": cc.OPUS}
        assert body == expected, _diff(expected, body)
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity, cc.OPUS, "PONG", {"input_tokens": 10, "output_tokens": 5}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{cc.OPUS}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            input_cost_per_token=1e-6,
            output_cost_per_token=5e-6,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(10 * 1e-6 + 5 * 5e-6)
