import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/converse/us.openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "gpt-5 reasoning wire control"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14},
        "metrics": {"latencyMs": 1},
    }
).encode()


def gpt5_converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/us.openai.gpt-5.6-sol/converse"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": [{"text": "synthetic reasoning request"}]}]
    assert body["additionalModelRequestFields"] == {"reasoning": {"effort": "high"}}, body
    assert body["inferenceConfig"] == {"maxTokens": 16}, body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_converse.gpt5_reasoning_effort_reaches_provider_as_reasoning_effort")
def test_gpt5_reasoning_effort_is_accepted_and_sent_as_converse_reasoning_effort(gateway: Gateway) -> None:
    with wire_server(gpt5_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL, api_key=TOKEN, aws_region_name="us-east-1", aws_bedrock_runtime_endpoint=wire.url
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic reasoning request"}],
                "reasoning_effort": "high",
                "max_tokens": 16,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "gpt-5 reasoning wire control", response.text
        assert response.json()["usage"]["total_tokens"] == 14, response.text
        assert len(wire.drain()) == 1
