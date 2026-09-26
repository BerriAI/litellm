import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_ADVISOR_KEY: Final = "synthetic-advisor-key"
_PROXY_ANTHROPIC_KEY: Final = "sk-proxy-owned-anthropic-secret"
_QUESTION: Final = "which index should this query use"
_ADVICE: Final = "use the composite index on (tenant_id, created_at)"
_FINAL_ANSWER: Final = "done, the composite index is the right one"


def _advisor_call_message(question: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "advisor-call",
                "type": "function",
                "function": {"name": "advisor", "arguments": json.dumps({"question": question})},
            }
        ],
    }


_FINAL_MESSAGE: Final = {"role": "assistant", "content": _FINAL_ANSWER}


def _chat_completion(identity: str, message: dict[str, object], finish_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{identity}",
                "object": "chat.completion",
                "created": 1,
                "model": "llama-3.3-70b-versatile",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        ).encode()
    )


def _executor_reply(body: dict[str, object], identity: str, question: str) -> Reply:
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    if any(message.get("role") == "tool" for message in messages):
        assert messages[-1]["content"] == _ADVICE
        return _chat_completion(identity, _FINAL_MESSAGE, "stop")
    tools: Final = body["tools"]
    assert isinstance(tools, list)
    assert tools[0]["function"]["name"] == "advisor"
    return _chat_completion(identity, _advisor_call_message(question), "tool_calls")


@pytest.mark.covers("providers.anthropic_messages_advisor.sub_call_uses_the_configured_advisor_deployment")
def test_advisor_sub_call_reaches_the_router_deployment_with_its_key_instead_of_anthropic_unauthenticated(
    gateway: Gateway,
) -> None:
    identity: Final = "advisor-wire-" + uuid.uuid4().hex
    migration: Final = "please plan the migration " + identity
    question: Final = _QUESTION + " " + identity

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        if request.target == "/v1/chat/completions":
            assert request.headers["authorization"] == "Bearer integration-provider-key"
            return _executor_reply(body, identity, question)
        assert request.target == "/v1/messages"
        assert request.headers["x-api-key"] == _ADVISOR_KEY
        assert body["model"] == "claude-opus-4-1-20250805"
        assert body["messages"] == [
            {"role": "user", "content": migration},
            {"role": "user", "content": question},
        ]
        assert "tools" not in body
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg-{identity}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-1-20250805",
                    "content": [{"type": "text", "text": _ADVICE}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 6},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        executor: Final = scenario.model(model="hosted_vllm/gpt-4o-mini", api_base=wire.url + "/v1")
        advisor: Final = scenario.model(
            model="anthropic/claude-opus-4-1-20250805", api_base=wire.url, api_key=_ADVISOR_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": executor,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": migration}],
                "tools": [{"type": "advisor_20260301", "name": "advisor", "model": advisor}],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["content"] == [{"type": "text", "text": _FINAL_ANSWER}], response.text
        assert body["stop_reason"] == "end_turn", response.text
        assert [request.target for request in wire.drain()] == [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/chat/completions",
        ]


def _advice_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"msg-{identity}",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-1-20250805",
                "content": [{"type": "text", "text": _ADVICE}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 6},
            }
        ).encode()
    )


@pytest.mark.covers("providers.anthropic_messages_advisor.caller_api_base_without_api_key_never_receives_the_proxy_key")
def test_advisor_api_base_without_api_key_is_rejected_before_the_proxy_anthropic_key_reaches_the_caller_host(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "advisor-leak-" + uuid.uuid4().hex
    question: Final = _QUESTION + " " + identity

    def executor(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        return _executor_reply(json.loads(request.body), identity, question)

    def caller_host(request: Request) -> Reply:
        return _advice_reply(identity)

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["allow_client_side_credentials"] = True
    path: Final = tmp_path / "client-side-credentials.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(executor) as executor_wire,
        wire_server(caller_host) as caller_wire,
        owned_proxy(gateway, tmp_path, {"ANTHROPIC_API_KEY": _PROXY_ANTHROPIC_KEY}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="hosted_vllm/gpt-4o-mini", api_base=executor_wire.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "please plan the migration"}],
                "tools": [
                    {
                        "type": "advisor_20260301",
                        "name": "advisor",
                        "model": "anthropic/claude-opus-4-1-20250805",
                        "api_base": caller_wire.url,
                    }
                ],
            },
        )
        received: Final = caller_wire.drain()
        assert [
            (request.target, request.headers.get("x-api-key"), json.loads(request.body)["messages"])
            for request in received
        ] == [], response.text
        assert response.is_error, response.text
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": (
                    "advisor tool definition sets 'api_base' without 'api_key'. A caller-supplied api_base is only "
                    "honored alongside a caller-supplied api_key, so the proxy's own credentials are never sent to a "
                    "caller-chosen destination."
                ),
            },
        }, response.text
        assert executor_wire.drain() == (), response.text
