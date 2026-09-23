import json
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.mcp import mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.observability.guardrails.rewrite_reaches_correct_anthropic_positions")
def test_guardrail_rewrites_system_and_user_in_actual_anthropic_request(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    originals: Final = ["synthetic private system", "synthetic private user", "unchanged sibling"]
    replacements: Final = ["permitted system", "permitted user", "unchanged sibling"]

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        body: Final = json.loads(request.body)
        assert body["texts"] == originals
        return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": replacements}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        body: Final = json.loads(request.body)
        assert body["system"] == [{"type": "text", "text": replacements[0]}]
        assert body["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": replacements[1]}, {"type": "text", "text": replacements[2]}],
            }
        ]
        assert all(text.encode() not in request.body for text in originals[:2])
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "permitted response"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "rewrite.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=upstream.url, api_key="synthetic-anthropic-key"
            )
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [
                        {"role": "system", "content": originals[0]},
                        {"role": "user", "content": [{"type": "text", "text": text} for text in originals[1:]]},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "permitted response"
            assert response.json()["choices"][0]["finish_reason"] == "stop"
            assert response.json()["usage"]["total_tokens"] == 15
            assert len(policy.drain()) == len(upstream.drain()) == 1


@pytest.mark.covers("other.observability.guardrails.denial_prevents_provider_with_allowed_control")
def test_guardrail_denial_prevents_provider_and_preserves_allowed_control(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        body: Final = json.loads(request.body)
        assert body["texts"] in (["synthetic denied marker"], ["synthetic allowed marker"])
        result: Final = (
            {"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}
            if body["texts"] == ["synthetic denied marker"]
            else {"action": "NONE"}
        )
        return Reply(body=json.dumps(result).encode())

    with wire_server(guardrail) as policy:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "deny.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            key: Final = scenario.key(models=[model])
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                denied: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "synthetic denied marker"}]},
                    key=key,
                )
                assert denied.status_code == 400 and "synthetic policy denial" in denied.text, denied.text
                assert observed.get("/__observations").json()["requests"] == []
                allowed: Final = candidate.chat(model, text="synthetic allowed marker", key=key)
                assert allowed["usage"]["total_tokens"] == 40
                assert (
                    allowed["choices"][0]["message"]["content"]
                    == "Hello! This is a mock response from the fake OpenAI endpoint."
                )
                assert len(observed.get("/__observations").json()["requests"]) == 1
            assert len(policy.drain()) == 2


@pytest.mark.covers("other.observability.guardrails.bedrock_passthrough_converse_scans_only_caller_content")
def test_bedrock_passthrough_converse_guardrail_ignores_denied_term_in_tool_definition(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    denied: Final = "synthetic denied marker"
    allowed: Final = "synthetic allowed weather question"
    access_key: Final = "AKIASYNTHETICPASSTHROUGH"
    tool_config: Final = {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup_weather",
                    "description": f"Look up the forecast, never answer a {denied}",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"city": {"type": "string", "enum": [denied]}},
                            "required": ["city"],
                        }
                    },
                }
            }
        ]
    }

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        texts: Final = json.loads(request.body)["texts"]
        result: Final = (
            {"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}
            if any(denied in text for text in texts)
            else {"action": "NONE"}
        )
        return Reply(body=json.dumps(result).encode())

    def runtime(request: Request) -> Reply:
        assert request.target == "/model/anthropic.claude-3-haiku-20240307-v1:0/converse"
        assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={access_key}/"), (
            request.headers
        )
        return Reply(
            body=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "sunny passthrough control"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(runtime) as bedrock, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            api_key=None,
            api_base=bedrock.url,
            aws_access_key_id=access_key,
            aws_secret_access_key="synthetic-secret",
            aws_region_name="us-east-1",
        )
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "bedrock-passthrough.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            route: Final = f"/bedrock/model/{model}/converse"
            passed: Final = candidate.request(
                "POST",
                route,
                {"messages": [{"role": "user", "content": [{"text": allowed}]}], "toolConfig": tool_config},
            )
            assert passed.status_code == 200, passed.text
            assert passed.json()["output"]["message"]["content"] == [{"text": "sunny passthrough control"}]
            forwarded: Final = bedrock.drain()
            assert len(forwarded) == 1, "the runtime peer must see exactly the allowed request"
            assert json.loads(forwarded[0].body)["toolConfig"] == tool_config
            blocked: Final = candidate.request(
                "POST",
                route,
                {"messages": [{"role": "user", "content": [{"text": denied}]}], "toolConfig": tool_config},
            )
            assert blocked.status_code == 400 and "synthetic policy denial" in blocked.text, blocked.text
            assert bedrock.drain() == ()
            assert [json.loads(request.body)["texts"] for request in policy.drain()] == [[allowed], [denied]]


@pytest.mark.covers("other.mcp.guardrails.request_selection_blocks_resolved_tool_without_execution")
def test_request_selected_mcp_guardrail_blocks_direct_and_virtual_calls(gateway: Gateway, tmp_path: Path) -> None:
    guardrail = "mcp-policy-" + uuid.uuid4().hex
    config = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": guardrail,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "pre_mcp_call",
                "default_on": False,
                "custom_code": (
                    "def apply_guardrail(inputs, request_data, input_type):\n"
                    '    if inputs.get("tools", [{}])[0].get("function", {}).get("name") == "add":\n'
                    '        return block("integration resolved add denied")\n'
                    "    return allow()\n"
                ),
            },
        }
    ]
    path = tmp_path / "mcp-guardrail.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        mcp_peer() as peer,
        candidate.scenario() as scenario,
    ):
        identity = register_mcp(scenario, peer, "guardrail" + uuid.uuid4().hex)
        permission = {"mcp_servers": [identity], "mcp_tool_search_enabled": True}
        key = scenario.key(object_permission=permission)
        key_selected = scenario.key(object_permission=permission, guardrails=[guardrail])
        team = scenario.team(guardrails=[guardrail], object_permission={"mcp_servers": [identity]})
        team_selected = scenario.key(team_id=team, object_permission=permission)
        catalog_key = scenario.key(object_permission={"mcp_servers": [identity]})
        names = tool_names(candidate, catalog_key, identity)
        assert set(names) == {"add", "multiply", "fail"}
        for virtual in (False, True):
            for caller, selected, tool, expected in (
                (key, [], "add", 8),
                (key, [guardrail], "add", None),
                (key_selected, [], "add", None),
                (team_selected, [], "add", None),
                (key, [guardrail], "multiply", 15),
            ):
                arguments = {"a": 3, "b": 5}
                peer.drain()
                response = candidate.client.post(
                    "/mcp-rest/tools/call",
                    headers={"x-litellm-api-key": caller},
                    json={
                        "server_id": identity,
                        "name": "mcp_tool_call" if virtual else names[tool],
                        "arguments": {"tool_name": names[tool], "arguments": arguments} if virtual else arguments,
                        "guardrails": selected,
                    },
                )
                calls = tuple(item for item in peer.drain() if item["body"].get("method") == "tools/call")
                if expected is None:
                    assert response.status_code == 400, response.text
                    assert "integration resolved add denied" in response.text, response.text
                    assert calls == (), "pre-call denial must prevent upstream execution"
                else:
                    assert response.status_code == 200, response.text
                    assert response.json()["isError"] is False
                    assert response.json()["content"][0]["text"] == str(expected), response.text
                    assert len(calls) == 1
                    assert calls[0]["body"]["params"]["name"] == tool
                    assert calls[0]["body"]["params"]["arguments"] == arguments


_RESPONSES_DENIAL: Final = "This model is not currently available."


def _responses_denial_config(tmp_path: Path, identity: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "pre_call",
                "default_on": False,
                "custom_code": (
                    f"def apply_guardrail(inputs, request_data, input_type):\n    return block({_RESPONSES_DENIAL!r})\n"
                ),
            },
        }
    ]
    path: Final = tmp_path / "responses-deny.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _assert_blocked_message_item(item: dict[str, object], response: dict[str, object]) -> None:
    assert item["type"] == "message", item
    assert item["role"] == "assistant", item
    assert item["status"] == "completed", item
    assert str(item["id"]).startswith("msg_"), item
    assert item["content"] == [{"type": "output_text", "text": _RESPONSES_DENIAL, "annotations": []}], item
    assert response["status"] == "completed", response
    usage: Final = response["usage"]
    assert isinstance(usage, dict), response
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (0, 0, 0), usage


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_streams_typed_message")
def test_responses_pre_call_denial_streams_sse_with_typed_message_item(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
            observed.get("/__observations")
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": "say hi", "stream": True, "guardrails": [identity]},
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream"), (
                response.headers["content-type"],
                response.text,
            )
            lines: Final = tuple(line for line in response.text.split("\n") if line.startswith("data: "))
            assert lines[-1] == "data: [DONE]", response.text
            events: Final = tuple(json.loads(line.removeprefix("data: ")) for line in lines[:-1])
            kinds: Final = tuple(event["type"] for event in events)
            assert tuple(kind for kind in kinds if kind != "response.output_text.delta") == (
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ), kinds
            assert kinds.index("response.output_text.delta") == kinds.index("response.content_part.added") + 1, kinds
            assert "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == (
                _RESPONSES_DENIAL
            )
            completed: Final = events[-1]["response"]
            assert completed["output"] == [events[-2]["item"]], (completed, events[-2])
            _assert_blocked_message_item(completed["output"][0], completed)
            assert observed.get("/__observations").json()["requests"] == []


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_returns_typed_message")
def test_responses_pre_call_denial_returns_json_with_typed_message_item(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
            observed.get("/__observations")
            response: Final = candidate.request(
                "POST", "/v1/responses", {"model": model, "input": "say hi", "guardrails": [identity]}
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("application/json"), response.headers["content-type"]
            body: Final = response.json()
            assert body["object"] == "response", body
            assert len(body["output"]) == 1, body
            _assert_blocked_message_item(body["output"][0], body)
            assert observed.get("/__observations").json()["requests"] == []
