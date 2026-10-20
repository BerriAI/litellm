import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.test_bedrock_auth_wire import MODEL, RESPONSE, TOKEN

ANTHROPIC_BETA: Final = ["interleaved-thinking-2025-05-14"]
CLIENT_METADATA: Final = {"originator": "codex_cli_rs", "version": "0.1.0", "session_id": "synthetic-session"}


def converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
    body: Final = json.loads(request.body)
    assert body["additionalModelRequestFields"] == {"anthropic_beta": ANTHROPIC_BETA}, body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_converse.client_metadata_is_not_forwarded_in_additional_model_request_fields")
def test_client_metadata_is_dropped_from_converse_body_while_anthropic_beta_is_kept(gateway: Gateway) -> None:
    with wire_server(converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic codex request"}],
                "max_tokens": 16,
                "anthropic_beta": ANTHROPIC_BETA,
                "client_metadata": CLIENT_METADATA,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "bedrock wire control"
        assert len(wire.drain()) == 1, response.text
