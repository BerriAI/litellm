import json
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/mantle/anthropic.claude-sonnet-5"
PROJECT_ID: Final = "synthetic-mantle-project"
TOKEN: Final = "synthetic-mantle-bearer"
RESPONSE: Final = json.dumps(
    {
        "id": "msg_mantle_project_control",
        "type": "message",
        "role": "assistant",
        "model": "anthropic.claude-sonnet-5",
        "content": [{"type": "text", "text": "mantle project control"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 9, "output_tokens": 3},
    }
).encode()


def project_scoped_peer(prompt: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/anthropic/v1/messages"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.headers.get("anthropic-workspace-id") == PROJECT_ID, dict(request.headers)
        assert "anthropic-workspace" not in request.headers, dict(request.headers)
        body: Final = json.loads(request.body)
        assert body["model"] == "anthropic.claude-sonnet-5"
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        assert body["max_tokens"] == 16
        assert not {"aws_bedrock_project_id", "timeout", "litellm_params", "litellm_metadata", "api_key"}.intersection(
            body
        )
        return Reply(body=RESPONSE)

    return respond


@pytest.mark.covers("other.provider_wire.bedrock.mantle_project_id_reaches_provider_as_workspace_id_header")
def test_mantle_project_id_is_sent_as_anthropic_workspace_id_header_on_chat_and_messages(gateway: Gateway) -> None:
    prompt: Final = f"synthetic project request {uuid.uuid4().hex}"
    with wire_server(project_scoped_peer(prompt)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            api_base=wire.url,
            aws_region_name="us-east-1",
            aws_bedrock_project_id=PROJECT_ID,
        )
        chat: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
            },
        )
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["message"]["content"] == "mantle project control"
        assert len(wire.drain()) == 1, "Expected exactly one chat completions call to reach the Mantle peer"
        messages: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            },
        )
        assert messages.status_code == 200, messages.text
        assert messages.json()["content"] == [{"type": "text", "text": "mantle project control"}]
        assert len(wire.drain()) == 1, "Expected exactly one messages call to reach the Mantle peer"
