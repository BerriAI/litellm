import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

R1_MODEL: Final = "bedrock/converse/us.deepseek.r1-v1:0"
V3_MODEL: Final = "bedrock/converse/deepseek.v3.2"
TOKEN: Final = "synthetic-bedrock-bearer"
RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "deepseek reasoning wire control"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14},
        "metrics": {"latencyMs": 1},
    }
).encode()


def r1_converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/us.deepseek.r1-v1%3A0/converse", request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": [{"text": "synthetic r1 request"}]}]
    assert body["inferenceConfig"] == {"maxTokens": 16}, body
    assert body.get("additionalModelRequestFields") is None, body
    return Reply(body=RESPONSE)


def v3_converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/deepseek.v3.2/converse", request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": [{"text": "synthetic v3 request"}]}]
    assert body["inferenceConfig"] == {"maxTokens": 16}, body
    assert body["additionalModelRequestFields"] == {"reasoning_effort": "high"}, body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_converse.deepseek_r1_drops_thinking_and_reasoning_effort_before_provider")
def test_deepseek_r1_thinking_and_reasoning_effort_are_dropped_instead_of_leaking_into_converse(
    gateway: Gateway,
) -> None:
    with wire_server(r1_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=R1_MODEL,
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            drop_params=True,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic r1 request"}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "reasoning_effort": "high",
                "max_tokens": 16,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "deepseek reasoning wire control", response.text
        assert response.json()["usage"]["total_tokens"] == 14, response.text
        assert len(wire.drain()) == 1


@pytest.mark.covers("providers.bedrock_converse.deepseek_v3_reasoning_effort_reaches_provider_raw")
def test_deepseek_v3_reasoning_effort_reaches_converse_raw_instead_of_as_anthropic_thinking(
    gateway: Gateway,
) -> None:
    with wire_server(v3_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=V3_MODEL, api_key=TOKEN, aws_region_name="us-east-1", aws_bedrock_runtime_endpoint=wire.url
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic v3 request"}],
                "reasoning_effort": "high",
                "max_tokens": 16,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "deepseek reasoning wire control", response.text
        assert response.json()["usage"]["total_tokens"] == 14, response.text
        assert len(wire.drain()) == 1
