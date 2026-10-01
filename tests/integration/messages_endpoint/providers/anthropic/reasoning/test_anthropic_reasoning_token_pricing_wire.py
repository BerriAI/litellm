import json
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


def test_reported_thinking_tokens_are_billed_at_the_reasoning_rate_and_the_rest_at_the_output_rate(
    gateway: Gateway,
) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [
                        {"type": "thinking", "thinking": "count the words", "signature": "EqQBCkgIBRABGAIiQLz"},
                        {"type": "text", "text": "PONG"},
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "output_tokens_details": {"thinking_tokens": 30},
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            input_cost_per_token=_INPUT_RATE,
            output_cost_per_token=_OUTPUT_RATE,
            output_cost_per_reasoning_token=_REASONING_RATE,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                **cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"),
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "stream": False,
                "model": model,
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
    assert float(rows[0]["spend"]) == pytest.approx(100 * _INPUT_RATE + 20 * _OUTPUT_RATE + 30 * _REASONING_RATE)
