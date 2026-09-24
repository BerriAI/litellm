import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.test_bedrock_auth_wire import MODEL, RESPONSE, TOKEN

GUARDRAIL: Final = {"guardrailIdentifier": "integration-guardrail", "guardrailVersion": "DRAFT", "trace": "enabled"}
PERFORMANCE: Final = {"latency": "optimized"}


def converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    body: Final = json.loads(request.body)
    assert body["inferenceConfig"] == {"maxTokens": 16, "temperature": 0.2}, body
    assert body["guardrailConfig"] == GUARDRAIL, body
    assert body["performanceConfig"] == PERFORMANCE, body
    return Reply(body=RESPONSE)


@pytest.mark.covers("other.provider_wire.bedrock.converse_config_blocks_sent_once_at_top_level")
def test_guardrail_and_performance_config_are_not_duplicated_inside_inference_config(gateway: Gateway) -> None:
    with wire_server(converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            guardrailConfig=GUARDRAIL,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic guardrail request"}],
                "max_tokens": 16,
                "temperature": 0.2,
                "performanceConfig": PERFORMANCE,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "bedrock wire control"
        assert len(wire.drain()) == 1, response.text
