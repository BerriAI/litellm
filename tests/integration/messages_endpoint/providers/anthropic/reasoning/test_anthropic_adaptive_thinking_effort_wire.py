import uuid
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.parametrize(
    ("upstream_model", "effort", "max_tokens"),
    (
        pytest.param(cc.FABLE, "high", 64000, id="fable-5.1-high"),
        pytest.param(cc.OPUS, "xhigh", 128000, id="opus-5.5-xhigh"),
    ),
)
def test_claude_code_adaptive_thinking_effort_and_reasoning_betas_reach_anthropic_intact(
    gateway: Gateway, upstream_model: str, effort: str, max_tokens: int
) -> None:
    request_body: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", effort, max_tokens)

    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(
                f"msg_{uuid.uuid4().hex}", upstream_model, "PONG", {"input_tokens": 12, "output_tokens": 4}
            ),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{upstream_model}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
    assert len(received) == 1, received
    expected: Final = {**request_body, "model": upstream_model}
    body: Final = cc.JSON_OBJECT.validate_json(received[0].body)
    assert body == expected, cc.body_diff(expected, body)
    betas: Final = cc.reasoning_betas(received[0].headers.get("anthropic-beta", ""))
    assert betas == cc.reasoning_betas(cc.FRONTIER_CLI_BETA), betas
