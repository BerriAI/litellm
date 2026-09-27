import json
import os
import signal
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.mcp import mcp_peer, register_mcp, tool_names
from integration._support.process import group_members, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from openai import AsyncOpenAI, OpenAI


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


@pytest.mark.covers("other.observability.guardrails.anthropic_messages_caller_metadata_keeps_guardrail_spend_log")
def test_anthropic_messages_with_caller_metadata_keeps_guardrail_information_in_spend_log(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    prompt: Final = "synthetic allowed prompt " + identity
    caller_metadata: Final = {"user_id": "device-account-session"}

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        assert json.loads(request.body)["texts"] == [prompt]
        return Reply(body=json.dumps({"action": "NONE"}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        body: Final = json.loads(request.body)
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert body["metadata"] == caller_metadata
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
        path: Final = tmp_path / "caller-metadata.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=upstream.url, api_key="synthetic-anthropic-key"
            )
            response: Final = candidate.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": prompt}],
                    "metadata": caller_metadata,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["content"] == [{"type": "text", "text": "permitted response"}], response.text
            assert response.headers["x-litellm-applied-guardrails"] == identity, dict(response.headers)
            assert len(policy.drain()) == len(upstream.drain()) == 1
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT call_type, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                    (model,),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert rows[0]["call_type"] == "anthropic_messages", rows[0]
            saved: Final = object_value(rows[0]["metadata"])
            entries: Final = saved["guardrail_information"]
            assert isinstance(entries, list) and len(entries) == 1, saved
            entry: Final = object_value(entries[0])
            assert entry["guardrail_name"] == identity, saved
            assert entry["guardrail_mode"] == "pre_call", saved
            assert entry["guardrail_status"] == "success", saved


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


@pytest.mark.covers("other.observability.guardrails.bedrock_post_call_scans_streamed_anthropic_messages_tool_use")
def test_bedrock_guardrail_streams_anthropic_messages_tool_use_instead_of_chunk_builder_500(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    spoken: Final = "Checking the forecast"
    frames: Final = (
        'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_synthetic", "type": "message", '
        '"role": "assistant", "model": "claude-sonnet-4-5-20250929", "content": [], "stop_reason": null, '
        '"stop_sequence": null, "usage": {"input_tokens": 11, "output_tokens": 1}}}\n\n',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, '
        '"content_block": {"type": "text", "text": ""}}\n\n',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
        f'"delta": {{"type": "text_delta", "text": "{spoken}"}}}}\n\n',
        'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 1, '
        '"content_block": {"type": "tool_use", "id": "toolu_synthetic", "name": "lookup_weather", "input": {}}}\n\n',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 1, '
        '"delta": {"type": "input_json_delta", "partial_json": "{\\"city\\": \\"Paris\\"}"}}\n\n',
        'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 1}\n\n',
        'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "tool_use", '
        '"stop_sequence": null}, "usage": {"output_tokens": 9}}\n\n',
        'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    )
    tools: Final = [
        {
            "name": "lookup_weather",
            "description": "Look up the forecast for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
    ]

    def guardrail(request: Request) -> Reply:
        assert request.target == f"/guardrail/{guardrail_id}/version/DRAFT/apply", request.target
        body: Final = json.loads(request.body)
        assert body["source"] == "OUTPUT", body
        assert body["content"] == [{"text": {"text": spoken}}], body
        return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        body: Final = json.loads(request.body)
        assert body["stream"] is True, body
        assert body["tools"] == tools, body
        return Reply(content_type="text/event-stream", chunks=tuple(frame.encode() for frame in frames))

    with wire_server(guardrail) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "bedrock",
                    "mode": "post_call",
                    "default_on": True,
                    "mask_response_content": True,
                    "guardrailIdentifier": guardrail_id,
                    "guardrailVersion": "DRAFT",
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
                    "aws_secret_access_key": "synthetic-secret",
                    "aws_bedrock_runtime_endpoint": policy.url,
                },
            }
        ]
        path: Final = tmp_path / "bedrock-stream.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=upstream.url, api_key="synthetic-anthropic-key"
            )
            response: Final = candidate.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 64,
                    "stream": True,
                    "tools": tools,
                    "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                },
            )
            assert response.status_code == 200, response.text
            head, separator, tail = response.text.partition("\n\n")
            assert separator == "\n\n", response.text
            assert head.startswith("event: message_start\ndata: "), response.text
            assert json.loads(head.removeprefix("event: message_start\ndata: ")) == {
                "type": "message_start",
                "message": {
                    "id": "msg_synthetic",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            }, response.text
            assert tail == "".join(frames[1:]), response.text
            assert len(policy.drain()) == len(upstream.drain()) == 1


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


def _deny_guardrail(name: str, denial: str = _RESPONSES_DENIAL) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "custom_code",
            "mode": "pre_call",
            "default_on": False,
            "custom_code": (f"def apply_guardrail(inputs, request_data, input_type):\n    return block({denial!r})\n"),
        },
    }


def _responses_denial_config(tmp_path: Path, identity: str, denial: str = _RESPONSES_DENIAL) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [_deny_guardrail(identity, denial)]
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


def _response_id(index: int, response: httpx.Response) -> str:
    assert response.status_code == 200, (index, response.text)
    if index % 3 == 0:
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        return str(_blocked_stream_events(response.text)[-1]["response"]["id"])
    if index % 3 == 1:
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        blocked: Final = _blocked_stream_events(response.text)[-1]["response"]
        _assert_blocked_message_item(blocked["output"][0], blocked)
        return str(blocked["id"])
    assert response.headers["content-type"].startswith("application/json"), response.text
    body: Final = response.json()
    _assert_blocked_message_item(body["output"][0], body)
    return str(body["id"])


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_streams_typed_message")
def test_responses_pre_call_denial_streams_sse_with_typed_message_item(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
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
        model: Final = scenario.model(use_chat_completions_api=True)
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


_RESPONSES_OUTPUT_DENIAL: Final = "Output withheld by policy."
_UPSTREAM_INPUT_TOKENS: Final = 20
_UPSTREAM_OUTPUT_TOKENS: Final = 20
_UPSTREAM_TOTAL_TOKENS: Final = 40


def _responses_output_denial_config(tmp_path: Path, identity: str, model: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "post_call",
                "default_on": False,
                "custom_code": (
                    "def apply_guardrail(inputs, request_data, input_type):\n"
                    f"    return block({_RESPONSES_OUTPUT_DENIAL!r})\n"
                ),
            },
        }
    ]
    config["policies"] = {
        f"{identity}-pipeline": {
            "guardrails": {"add": [identity]},
            "pipeline": {
                "mode": "post_call",
                "steps": [
                    {
                        "guardrail": identity,
                        "on_pass": "allow",
                        "on_fail": "modify_response",
                        "modify_response_message": _RESPONSES_OUTPUT_DENIAL,
                    }
                ],
            },
        }
    }
    config["policy_attachments"] = [{"policy": f"{identity}-pipeline", "models": [model]}]
    path: Final = tmp_path / "responses-output-deny.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _blocked_stream_events(text: str) -> tuple[dict[str, object], ...]:
    lines: Final = tuple(line for line in text.split("\n") if line.startswith("data: "))
    assert lines[-1] == "data: [DONE]", text
    return tuple(json.loads(line.removeprefix("data: ")) for line in lines[:-1])


def _dead_api_base() -> str:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = reserve.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_openai_sdk_streams_typed_message")
def test_responses_pre_call_denial_openai_sdk_streams_typed_message(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        client: Final = OpenAI(
            base_url=f"{candidate.client.base_url}/v1", api_key=candidate.key, max_retries=0, timeout=15
        )
        events: Final = tuple(
            client.responses.create(model=model, input="say hi", stream=True, extra_body={"guardrails": [identity]})
        )
        assert events[-1].type == "response.completed", [event.type for event in events]
        completed: Final = events[-1].response
        assert completed is not None and len(completed.output) == 1, completed
        item: Final = completed.output[0]
        assert item.type == "message", item
        assert item.role == "assistant" and item.status == "completed", item
        assert item.content[0].type == "output_text" and item.content[0].text == _RESPONSES_DENIAL, item.content
        assert completed.usage is not None and completed.usage.total_tokens == 0, completed.usage


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_openai_async_sdk_streams_typed_message")
async def test_responses_pre_call_denial_openai_async_sdk_streams_typed_message(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        client: Final = AsyncOpenAI(
            base_url=f"{candidate.client.base_url}/v1", api_key=candidate.key, max_retries=0, timeout=15
        )
        stream: Final = await client.responses.create(
            model=model, input="say hi", stream=True, extra_body={"guardrails": [identity]}
        )
        kinds: Final = [event.type async for event in stream]
        assert kinds[-1] == "response.completed", kinds
        assert "response.output_text.delta" in kinds, kinds
        assert "response.in_progress" in kinds, kinds


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_openai_sdk_returns_typed_message")
def test_responses_pre_call_denial_openai_sdk_returns_typed_message(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        client: Final = OpenAI(
            base_url=f"{candidate.client.base_url}/v1", api_key=candidate.key, max_retries=0, timeout=15
        )
        body: Final = client.responses.create(model=model, input="say hi", extra_body={"guardrails": [identity]})
        assert body.object == "response" and body.status == "completed", body
        assert len(body.output) == 1, body.output
        item: Final = body.output[0]
        assert item.type == "message" and item.role == "assistant", item
        assert item.content[0].type == "output_text" and item.content[0].text == _RESPONSES_DENIAL, item.content
        assert body.output_text == _RESPONSES_DENIAL, body
        assert body.usage is not None and body.usage.total_tokens == 0, body.usage


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_false_returns_json")
def test_responses_pre_call_denial_stream_false_returns_json(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        response: Final = candidate.request(
            "POST", "/v1/responses", {"model": model, "input": "say hi", "stream": False, "guardrails": [identity]}
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json"), response.text
        body: Final = response.json()
        _assert_blocked_message_item(body["output"][0], body)


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_string_true_returns_json")
def test_responses_pre_call_denial_stream_string_true_returns_json(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        response: Final = candidate.request(
            "POST", "/v1/responses", {"model": model, "input": "say hi", "stream": "true", "guardrails": [identity]}
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json"), (
            response.headers["content-type"],
            response.text,
        )
        body: Final = response.json()
        _assert_blocked_message_item(body["output"][0], body)


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_event_vocabulary")
def test_responses_pre_call_denial_stream_event_vocabulary(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    second: Final = "guardrail-2-" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    loaded: Final = yaml.safe_load(config.read_text())
    loaded["guardrails"].append(_deny_guardrail(second))
    config.write_text(yaml.safe_dump(loaded))
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
            observed.get("/__observations")
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": "say hi", "stream": True, "guardrails": [identity, second]},
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream"), response.text
            events: Final = _blocked_stream_events(response.text)
            kinds: Final = {event["type"] for event in events}
            assert kinds == {
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            }, kinds
            item_done: Final = tuple(event for event in events if event["type"] == "response.output_item.done")
            assert len(item_done) == 1, events
            assert len(events[-1]["response"]["output"]) == 1, events[-1]
            assert observed.get("/__observations").json()["requests"] == []


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_large_denial_text")
def test_responses_pre_call_denial_stream_large_denial_text(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    denial: Final = ("Denied: " + "mixed ascii and unicode text " * 200 + "fin")[:5000]
    config: Final = _responses_denial_config(tmp_path, identity, denial)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": "say hi", "stream": True, "guardrails": [identity]},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        events: Final = _blocked_stream_events(response.text)
        assert "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == denial
        done: Final = next(event for event in events if event["type"] == "response.output_text.done")
        assert done["text"] == denial, done
        completed: Final = events[-1]["response"]
        assert completed["output"][0]["content"][0]["text"] == denial, completed


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_requests_have_distinct_ids")
def test_responses_pre_call_denial_stream_requests_have_distinct_ids(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(use_chat_completions_api=True)
        with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
            observed.get("/__observations")
            responses: Final = tuple(
                candidate.request(
                    "POST",
                    "/v1/responses",
                    {"model": model, "input": "say hi", "stream": True, "guardrails": [identity]},
                )
                for _ in range(2)
            )
            completed: Final = tuple(_blocked_stream_events(response.text)[-1]["response"] for response in responses)
            for response in responses:
                assert response.status_code == 200, response.text
                assert response.headers["content-type"].startswith("text/event-stream"), response.text
            assert completed[0]["id"] != completed[1]["id"], completed
            assert completed[0]["output"][0]["id"] != completed[1]["output"][0]["id"], completed
            assert observed.get("/__observations").json()["requests"] == []


def _register_named_model(candidate: Gateway, name: str, api_base: str | None = None, **parameters: object) -> str:
    created: Final = candidate.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": api_base or f"{candidate.upstream_url}/v1",
                **parameters,
            },
        },
    )
    return str(created["model_info"]["id"])


@pytest.mark.covers("other.observability.guardrails.responses_post_call_pipeline_denial_streams_real_usage")
def test_responses_post_call_pipeline_denial_streams_real_usage(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    model: Final = f"integration-{uuid.uuid4().hex}"
    config: Final = _responses_output_denial_config(tmp_path, identity, model)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
        model_id: Final = _register_named_model(candidate, model, use_chat_completions_api=True)
        try:
            response: Final = candidate.request(
                "POST", "/v1/responses", {"model": model, "input": f"say hi {uuid.uuid4().hex}", "stream": True}
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream"), response.text
            events: Final = _blocked_stream_events(response.text)
            assert events[-1]["type"] == "response.completed", events
            completed: Final = events[-1]["response"]
            item: Final = completed["output"][0]
            assert item["type"] == "message" and item["role"] == "assistant", item
            assert item["content"][0]["type"] == "output_text", item
            assert item["content"][0]["text"] == _RESPONSES_OUTPUT_DENIAL, item
            usage: Final = completed["usage"]
            assert (
                usage["input_tokens"],
                usage["output_tokens"],
                usage["total_tokens"],
            ) == (_UPSTREAM_INPUT_TOKENS, _UPSTREAM_OUTPUT_TOKENS, _UPSTREAM_TOTAL_TOKENS), usage
        finally:
            candidate.post("/model/delete", {"id": model_id})


@pytest.mark.covers("other.observability.guardrails.responses_post_call_pipeline_denial_returns_real_usage")
def test_responses_post_call_pipeline_denial_returns_real_usage(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    model: Final = f"integration-{uuid.uuid4().hex}"
    config: Final = _responses_output_denial_config(tmp_path, identity, model)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
        model_id: Final = _register_named_model(candidate, model, use_chat_completions_api=True)
        try:
            response: Final = candidate.request(
                "POST", "/v1/responses", {"model": model, "input": f"say hi {uuid.uuid4().hex}"}
            )
            assert response.status_code == 200, response.text
            body: Final = response.json()
            item: Final = body["output"][0]
            assert item["type"] == "message" and item["role"] == "assistant", item
            assert item["content"][0]["type"] == "output_text", item
            assert item["content"][0]["text"] == _RESPONSES_OUTPUT_DENIAL, item
            usage: Final = body["usage"]
            assert (
                usage["input_tokens"],
                usage["output_tokens"],
                usage["total_tokens"],
            ) == (_UPSTREAM_INPUT_TOKENS, _UPSTREAM_OUTPUT_TOKENS, _UPSTREAM_TOTAL_TOKENS), usage
        finally:
            candidate.post("/model/delete", {"id": model_id})


@pytest.mark.covers("other.observability.guardrails.responses_denial_requires_authentication")
def test_responses_denial_requires_authentication(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = candidate.request(
            "POST", "/v1/responses", {"model": model, "input": "say hi", "guardrails": [identity]}, key="sk-invalid"
        )
        assert response.status_code == 401, (response.status_code, response.text)
        assert response.json()["error"]["type"] == "token_not_found_in_db", response.text


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_does_not_reach_upstream")
def test_responses_pre_call_denial_stream_does_not_reach_upstream(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(api_base=_dead_api_base())
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": "say hi", "stream": True, "guardrails": [identity]},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        events: Final = _blocked_stream_events(response.text)
        completed: Final = events[-1]["response"]
        _assert_blocked_message_item(completed["output"][0], completed)


@pytest.mark.covers("other.observability.guardrails.responses_unguarded_stream_reaches_upstream")
def test_responses_unguarded_stream_reaches_upstream(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        dead: Final = scenario.model(api_base=_dead_api_base())
        denied: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": dead, "input": "say hi", "stream": True, "guardrails": [identity]},
        )
        assert denied.status_code == 200, denied.text
        model: Final = scenario.model(use_chat_completions_api=True)
        with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
            observed.get("/__observations")
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": f"say hi {uuid.uuid4().hex}", "stream": True, "guardrails": []},
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream"), response.text
            assert "response.completed" in response.text, response.text
            requests: Final = eventually(
                lambda: observed.get("/__observations").json()["requests"],
                lambda values: len(values) >= 1,
                seconds=30,
            )
            assert len(requests) == 1, requests
            assert requests[0]["path"] == "/v1/chat/completions", requests


@pytest.mark.covers("other.observability.guardrails.chat_pre_call_denial_streams_content_filter")
def test_chat_pre_call_denial_streams_content_filter(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "say hi"}],
                "stream": True,
                "guardrails": [identity],
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        lines: Final = tuple(line for line in response.text.split("\n") if line.startswith("data: "))
        assert lines[-1] == "data: [DONE]", response.text
        chunks: Final = tuple(json.loads(line.removeprefix("data: ")) for line in lines[:-1])
        assert chunks[0]["choices"][0]["delta"]["content"] == _RESPONSES_DENIAL, chunks
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks


@pytest.mark.covers("other.observability.guardrails.chat_pre_call_denial_returns_content_filter")
def test_chat_pre_call_denial_returns_content_filter(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "say hi"}],
                "guardrails": [identity],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        choice: Final = body["choices"][0]
        assert choice["finish_reason"] == "content_filter", body
        assert choice["message"]["content"] == _RESPONSES_DENIAL, body
        assert (
            body["usage"]["prompt_tokens"],
            body["usage"]["completion_tokens"],
            body["usage"]["total_tokens"],
        ) == (0, 0, 0), body["usage"]


@pytest.mark.covers("other.observability.guardrails.messages_pre_call_denial_returns_message")
def test_messages_pre_call_denial_returns_message(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 16,
                "guardrails": [identity],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["type"] == "message" and body["role"] == "assistant", body
        assert body["content"] == [{"type": "text", "text": _RESPONSES_DENIAL}], body
        assert body["stop_reason"] == "end_turn", body
        assert (body["usage"]["input_tokens"], body["usage"]["output_tokens"]) == (0, 0), body


@pytest.mark.covers("other.observability.guardrails.messages_pre_call_denial_streams_message")
def test_messages_pre_call_denial_streams_message(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 16,
                "stream": True,
                "guardrails": [identity],
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        lines: Final = tuple(line for line in response.text.split("\n") if line.startswith("data: "))
        assert len(lines) == 1, response.text
        body: Final = json.loads(lines[0].removeprefix("data: "))
        assert body["type"] == "message" and body["role"] == "assistant", body
        assert body["content"] == [{"type": "text", "text": _RESPONSES_DENIAL}], body


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_writes_zero_spend_row")
def test_responses_pre_call_denial_writes_zero_spend_row(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = candidate.request(
            "POST", "/v1/responses", {"model": model, "input": "say hi", "guardrails": [identity]}
        )
        assert response.status_code == 200, response.text
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, total_tokens FROM "LiteLLM_SpendLogs" WHERE model=%s AND call_type=%s',
                (model, "aresponses"),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == 0, rows
        assert rows[0]["total_tokens"] == 0, rows


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_survives_worker_burst")
def test_responses_pre_call_denial_stream_survives_worker_burst(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model(api_base=_dead_api_base())
        healthy: Final = scenario.model(use_chat_completions_api=True)
        with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
            observed.get("/__observations")

            def burst(index: int) -> httpx.Response:
                if index % 3 == 0:
                    return candidate.request(
                        "POST",
                        "/v1/responses",
                        {"model": healthy, "input": f"say hi {uuid.uuid4().hex} {index}", "stream": True},
                    )
                stream: Final = index % 3 == 1
                return candidate.request(
                    "POST",
                    "/v1/responses",
                    {"model": model, "input": f"say hi {index}", "stream": stream, "guardrails": [identity]},
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                responses: Final = tuple(pool.map(burst, range(30)))
            response_ids: Final = frozenset(_response_id(index, response) for index, response in enumerate(responses))
            assert len(response_ids) == 30, response_ids
            assert len(observed.get("/__observations").json()["requests"]) == 10


@pytest.mark.covers("other.observability.guardrails.responses_pre_call_denial_stream_survives_worker_kill")
def test_responses_pre_call_denial_stream_survives_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    config: Final = _responses_denial_config(tmp_path, identity)
    with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=_dead_api_base())
            members: Final = tuple(
                member for member in group_members(owned.process.pid) if member.pid != owned.process.pid
            )
            children: Final = tuple(member.pid for member in members)
            workers: Final = tuple(
                member.pid for member in members if any("spawn_main" in part for part in member.cmdline())
            )
            assert len(workers) >= 2, workers
            os.kill(workers[0], signal.SIGKILL)
            expected: Final = len(children)
            eventually(
                lambda: tuple(
                    member.pid
                    for member in group_members(owned.process.pid)
                    if member.pid != owned.process.pid
                    and member.is_running()
                    and member.status() != psutil.STATUS_ZOMBIE
                ),
                lambda pids: len(pids) >= expected and any(pid not in children for pid in pids),
                seconds=30,
            )

            def burst(index: int) -> httpx.Response:
                return candidate.request(
                    "POST",
                    "/v1/responses",
                    {"model": model, "input": f"say hi {index}", "stream": True, "guardrails": [identity]},
                )

            with ThreadPoolExecutor(max_workers=5) as pool:
                responses: Final = tuple(pool.map(burst, range(10)))
            for response in responses:
                assert response.status_code == 200, response.text
                assert response.headers["content-type"].startswith("text/event-stream"), response.text
