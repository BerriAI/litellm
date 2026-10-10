import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/invoke/us.anthropic.claude-opus-4-8"
TOKEN: Final = "synthetic-bedrock-bearer"
RESPONSE: Final = json.dumps(
    {
        "id": "msg_adaptive_control",
        "type": "message",
        "role": "assistant",
        "model": "us.anthropic.claude-opus-4-8",
        "content": [{"type": "text", "text": "adaptive thinking control"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }
).encode()


def adaptive_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/us.anthropic.claude-opus-4-8/invoke"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "synthetic effort request"}]}]
    assert body["thinking"]["type"] == "adaptive", body
    assert body["output_config"] == {"effort": "high"}, body
    assert "budget_tokens" not in json.dumps(body), body
    return Reply(body=RESPONSE)


@pytest.mark.covers("other.provider_wire.bedrock.prefixed_opus_4_8_reasoning_effort_sends_adaptive_thinking")
def test_prefixed_opus_4_8_reasoning_effort_reaches_bedrock_as_adaptive_thinking_not_budget_tokens(
    gateway: Gateway,
) -> None:
    with wire_server(adaptive_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic effort request"}],
                "max_tokens": 4096,
                "reasoning_effort": "high",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "adaptive thinking control"
        assert response.json()["usage"]["prompt_tokens"] == 12 and response.json()["usage"]["completion_tokens"] == 5
        assert len(wire.drain()) == 1
