import uuid
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "claude-haiku-4-5"
_INPUT_RATE: Final = 1e-6
_OUTPUT_RATE: Final = 2e-6
_REASONING_RATE: Final = 7e-6
_CONTENT: Final = (
    {"type": "thinking", "thinking": "count the words", "signature": "EqQBCkgIBRABGAIiQLz"},
    {"type": "text", "text": "PONG"},
)
_USAGE: Final = {"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"thinking_tokens": 30}}


def _reply(identity: str, stream: bool) -> Reply:
    if stream:
        return Reply(chunks=cc.message_stream(identity, _MODEL, _CONTENT, _USAGE), content_type="text/event-stream")
    return Reply(body=cc.message_reply(identity, _MODEL, _CONTENT, _USAGE))


@pytest.mark.parametrize("stream", (pytest.param(False, id="non-streamed"), pytest.param(True, id="streamed")))
def test_reported_thinking_tokens_are_billed_at_the_reasoning_rate_and_the_rest_at_the_output_rate(
    gateway: Gateway, stream: bool
) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _reply(identity, stream)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            input_cost_per_token=_INPUT_RATE,
            output_cost_per_token=_OUTPUT_RATE,
            output_cost_per_reasoning_token=_REASONING_RATE,
        )
        body: Final = {
            **cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"),
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "stream": stream,
            "model": model,
        }
        response: Final = gateway.request("POST", "/v1/messages", body)
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
    input_tokens, output_tokens, thinking_tokens = 100, 50, 30
    assert float(rows[0]["spend"]) == pytest.approx(
        input_tokens * _INPUT_RATE
        + (output_tokens - thinking_tokens) * _OUTPUT_RATE
        + thinking_tokens * _REASONING_RATE
    ), rows
