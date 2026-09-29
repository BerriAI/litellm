import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration._support import claude_code as cc

_BETA_1M: Final = f"{cc.FRONTIER_CLI_BETA.replace(',effort-2025-11-24', ',context-1m-2025-08-07,effort-2025-11-24')}"


def test_1m_context_beta_forwarded_and_tiered_prompt_priced_above_200k(gateway: Gateway) -> None:
    identity: Final = f"msg_1m_{uuid.uuid4().hex}"
    request_body: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000)

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        upstream_beta: Final = request.headers.get("anthropic-beta", "")
        assert upstream_beta.split(",").count("context-1m-2025-08-07") == 1, upstream_beta
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity, cc.FABLE, "PONG", {"input_tokens": 250000, "output_tokens": 100}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{cc.FABLE}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            input_cost_per_token=1e-6,
            input_cost_per_token_above_200k_tokens=2e-6,
            output_cost_per_token=5e-6,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, _BETA_1M),
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
        assert float(rows[0]["spend"]) == pytest.approx(250000 * 2e-6 + 100 * 5e-6), dict(rows[0])
