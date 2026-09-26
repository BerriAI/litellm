import json
import re
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.mcp import mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PNG_BASE64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
PDF_BASE64: Final = "JVBERi0xLjQKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PgplbmRvYmoKMyAwIG9iago8PC9UeXBlL1BhZ2UvUGFyZW50IDIgMCBSL01lZGlhQm94WzAgMCA2MTIgNzkyXT4+CmVuZG9iago="


def _bedrock_policy(endpoint: str) -> dict[str, JsonValue]:
    return {
        "guardrail": "bedrock",
        "mode": "pre_call",
        "default_on": True,
        "guardrailIdentifier": "synthetic-guardrail",
        "guardrailVersion": "1",
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "synthetic-access-key",
        "aws_secret_access_key": "synthetic-secret-key",
        "aws_bedrock_runtime_endpoint": endpoint,
    }


def _guardrail_config(tmp_path: Path, name: str, params: dict[str, JsonValue], filename: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [{"guardrail_name": name, "litellm_params": params}]
    path: Final = tmp_path / filename
    path.write_text(yaml.safe_dump(config))
    return path


def _allow(_request: Request) -> Reply:
    return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')


def _unreachable(_request: Request) -> Reply:
    return Reply(status=500)


def _anthropic_reply(request: Request) -> Reply:
    assert request.target == "/v1/messages"
    return Reply(
        body=json.dumps(
            {
                "id": "wire-message",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": "wire response"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }
        ).encode()
    )


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


def test_bedrock_guardrail_scans_image_only_chat_request(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    def guardrail(request: Request) -> Reply:
        assert request.target == "/guardrail/synthetic-guardrail/version/1/apply"
        body: Final = json.loads(request.body)
        assert body["content"] == [{"image": {"format": "png", "source": {"bytes": PNG_BASE64}}}]
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    with wire_server(guardrail) as policy:
        path: Final = _guardrail_config(tmp_path, identity, _bedrock_policy(policy.url), "bedrock-image.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"}}
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 200, response.text
                assert len(observed.get("/__observations").json()["requests"]) == 1
            assert len(policy.drain()) == 1


def test_bedrock_guardrail_refuses_chat_file_part_without_reaching_guardrail_or_provider(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    with wire_server(_allow) as policy:
        path: Final = _guardrail_config(tmp_path, identity, _bedrock_policy(policy.url), "bedrock-file.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "synthetic summarize request"},
                                    {
                                        "type": "file",
                                        "file": {
                                            "file_data": f"data:application/pdf;base64,{PDF_BASE64}",
                                            "filename": "a.pdf",
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "document/file attachment(s) cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


def test_bedrock_guardrail_refuses_anthropic_document_block(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    with wire_server(_allow) as policy, wire_server(_anthropic_reply) as upstream:
        path: Final = _guardrail_config(tmp_path, identity, _bedrock_policy(policy.url), "bedrock-document.yaml")
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
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "synthetic summarize request"},
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": PDF_BASE64,
                                    },
                                },
                            ],
                        }
                    ],
                },
            )
            assert response.status_code == 400, response.text
            assert "document/file attachment(s) cannot be scanned" in response.text, response.text
            assert policy.drain() == upstream.drain() == ()


def test_bedrock_guardrail_refuses_responses_input_file(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    with wire_server(_allow) as policy:
        path: Final = _guardrail_config(tmp_path, identity, _bedrock_policy(policy.url), "bedrock-input-file.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/responses",
                    {
                        "model": model,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "synthetic summarize request"},
                                    {
                                        "type": "input_file",
                                        "file_data": f"data:application/pdf;base64,{PDF_BASE64}",
                                        "filename": "a.pdf",
                                    },
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "document/file attachment(s) cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


def test_bedrock_guardrail_text_only_request_payload_is_texts_only(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    def guardrail(request: Request) -> Reply:
        assert request.target == "/guardrail/synthetic-guardrail/version/1/apply"
        body: Final = json.loads(request.body)
        assert body["content"] == [{"text": {"text": "synthetic text only request"}}]
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    with wire_server(guardrail) as policy:
        path: Final = _guardrail_config(tmp_path, identity, _bedrock_policy(policy.url), "bedrock-text.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "synthetic text only request"}]},
            )
            assert response.status_code == 200, response.text
            assert len(policy.drain()) == 1


def test_non_bedrock_guardrail_is_not_invoked_on_image_only_request(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex

    with wire_server(_unreachable) as policy:
        params: Final = {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": True,
            "api_base": policy.url,
            "api_key": "synthetic-guardrail-key",
        }
        path: Final = _guardrail_config(tmp_path, identity, params, "generic-image.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"}}
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 200, response.text
                assert len(observed.get("/__observations").json()["requests"]) == 1
            assert policy.drain() == ()


_SSN_PATTERN: Final = re.compile(r"\d{3}-\d{2}-\d{4}")


def _masked_policy(_request: Request) -> Reply:
    body: Final = json.loads(_request.body)
    outputs: Final = [
        {"text": _SSN_PATTERN.sub("{SSN}", item["text"]["text"])} for item in body["content"] if "text" in item
    ]
    return Reply(
        body=json.dumps(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": outputs,
                "assessments": [
                    {
                        "sensitiveInformationPolicy": {
                            "piiEntities": [{"type": "US_SSN", "match": "x", "action": "ANONYMIZED"}]
                        }
                    }
                ],
            }
        ).encode()
    )


def _during_call_masking_config(tmp_path: Path, policy_url: str, check: str, filename: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    bedrock_params: Final = _bedrock_policy(policy_url)
    bedrock_params["mode"] = "during_call"
    bedrock_params["mask_request_content"] = True
    config["guardrails"] = [
        {"guardrail_name": "bedrock-during-" + uuid.uuid4().hex, "litellm_params": bedrock_params},
        {
            "guardrail_name": "mask-check-" + uuid.uuid4().hex,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "post_call",
                "default_on": True,
                "custom_code": check,
            },
        },
    ]
    path: Final = tmp_path / filename
    path.write_text(yaml.safe_dump(config))
    return path


def test_bedrock_during_call_mask_keeps_tool_result(gateway: Gateway, tmp_path: Path) -> None:
    check: Final = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        "    flat = str(request_data)\n"
        '    if "record SSN {SSN}" in flat and "please summarise" in flat and "123-45-6789" not in flat:\n'
        "        return allow()\n"
        '    return block("mask check failed")'
    )

    with wire_server(_masked_policy) as policy, wire_server(_anthropic_reply) as upstream:
        path: Final = _during_call_masking_config(tmp_path, policy.url, check, "during-mask.yaml")
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
                    "messages": [
                        {"role": "user", "content": "q"},
                        {
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "toolu_01A", "name": "lookup", "input": {}}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_01A",
                                    "content": "record SSN 123-45-6789",
                                },
                                {"type": "text", "text": "please summarise"},
                            ],
                        },
                    ],
                },
            )
            assert response.status_code == 200, response.text
            eventually(lambda: policy.drain(), lambda values: len(values) == 1, seconds=10)
            eventually(lambda: upstream.drain(), lambda values: len(values) == 1, seconds=10)


def test_bedrock_during_call_mask_survives_pii_in_text(gateway: Gateway, tmp_path: Path) -> None:
    check: Final = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        "    flat = str(request_data)\n"
        '    if "no pii here" in flat and "my ssn {SSN}" in flat and "123-45-6789" not in flat:\n'
        "        return allow()\n"
        '    return block("mask check failed")'
    )

    with wire_server(_masked_policy) as policy, wire_server(_anthropic_reply) as upstream:
        path: Final = _during_call_masking_config(tmp_path, policy.url, check, "during-mask2.yaml")
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
                    "messages": [
                        {"role": "user", "content": "q"},
                        {
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "toolu_01A", "name": "lookup", "input": {}}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "toolu_01A", "content": "no pii here"},
                                {"type": "text", "text": "my ssn 123-45-6789"},
                            ],
                        },
                    ],
                },
            )
            assert response.status_code == 200, response.text
            eventually(lambda: policy.drain(), lambda values: len(values) == 1, seconds=10)
            eventually(lambda: upstream.drain(), lambda values: len(values) == 1, seconds=10)


def test_non_bedrock_policy_inputs_match_base_on_empty_tools(gateway: Gateway, tmp_path: Path) -> None:
    def sink(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    with wire_server(sink) as policy:
        params: Final = {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": True,
            "api_base": policy.url,
            "api_key": "synthetic-guardrail-key",
        }
        path: Final = _guardrail_config(tmp_path, "generic-tools-" + uuid.uuid4().hex, params, "generic-tools.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "hi there"}], "tools": []},
            )
            assert response.status_code == 200, response.text
            sink_requests: Final = policy.drain()
            assert len(sink_requests) == 1
            assert json.loads(sink_requests[0].body)["tools"] is None, sink_requests[0].body


def test_bedrock_refuses_responses_function_call_output_file(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_allow) as policy:
        path: Final = _guardrail_config(
            tmp_path, "bedrock-fco-" + uuid.uuid4().hex, _bedrock_policy(policy.url), "bedrock-fco.yaml"
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/responses",
                    {
                        "model": model,
                        "input": [
                            {
                                "type": "function_call_output",
                                "call_id": "c1",
                                "output": [
                                    {
                                        "type": "input_file",
                                        "filename": "a.pdf",
                                        "file_data": f"data:application/pdf;base64,{PDF_BASE64}",
                                    }
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "document/file attachment(s) cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


def test_bedrock_scans_responses_function_call_output_image(gateway: Gateway, tmp_path: Path) -> None:
    def guardrail(request: Request) -> Reply:
        assert request.target == "/guardrail/synthetic-guardrail/version/1/apply"
        body: Final = json.loads(request.body)
        assert body["content"] == [{"image": {"format": "png", "source": {"bytes": PNG_BASE64}}}]
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    def responses_reply(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "resp-wire",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "m1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "wire response", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(responses_reply) as upstream:
        path: Final = _guardrail_config(
            tmp_path, "bedrock-fcoi-" + uuid.uuid4().hex, _bedrock_policy(policy.url), "bedrock-fcoi.yaml"
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4o-mini", api_base=f"{upstream.url}/v1", api_key="synthetic-openai-key"
            )
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "c1",
                            "output": [{"type": "input_image", "image_url": f"data:image/png;base64,{PNG_BASE64}"}],
                        }
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert len(upstream.drain()) == 1
            assert len(policy.drain()) == 1


def test_bedrock_refuses_chat_video_part(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_allow) as policy:
        path: Final = _guardrail_config(
            tmp_path, "bedrock-video-" + uuid.uuid4().hex, _bedrock_policy(policy.url), "bedrock-video.yaml"
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "look"},
                                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


def test_bedrock_during_call_refuses_chat_video_part(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_allow) as policy:
        params: Final = _bedrock_policy(policy.url)
        params["mode"] = "during_call"
        path: Final = _guardrail_config(
            tmp_path, "bedrock-during-video-" + uuid.uuid4().hex, params, "bedrock-during-video.yaml"
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "look"},
                                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


def test_bedrock_during_call_refuses_responses_function_call_output_file(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_allow) as policy:
        params: Final = _bedrock_policy(policy.url)
        params["mode"] = "during_call"
        path: Final = _guardrail_config(tmp_path, "bedrock-dfco-" + uuid.uuid4().hex, params, "bedrock-dfco.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/responses",
                    {
                        "model": model,
                        "input": [
                            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
                            {
                                "type": "function_call_output",
                                "call_id": "c1",
                                "output": [
                                    {
                                        "type": "input_file",
                                        "filename": "a.pdf",
                                        "file_data": f"data:application/pdf;base64,{PDF_BASE64}",
                                    }
                                ],
                            },
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "hi"}],
                            },
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


@pytest.mark.parametrize(
    "media_part",
    [
        {"type": "video_url", "video_url": {"url": "https://synthetic.example/clip.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ],
)
def test_bedrock_latest_role_during_call_refuses_scoped_media(
    gateway: Gateway, tmp_path: Path, media_part: dict[str, JsonValue]
) -> None:
    with wire_server(_allow) as policy:
        params: Final = _bedrock_policy(policy.url)
        params["mode"] = "during_call"
        params["experimental_use_latest_role_message_only"] = True
        path: Final = _guardrail_config(tmp_path, "bedrock-latdur-" + uuid.uuid4().hex, params, "bedrock-latdur.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "watch this"}, media_part]},
                            {"role": "assistant", "content": "ok"},
                            {"role": "user", "content": "next"},
                        ],
                    },
                )
                assert response.status_code == 400, response.text
                assert "cannot be scanned" in response.text, response.text
                assert observed.get("/__observations").json()["requests"] == []
            assert policy.drain() == ()


@pytest.mark.parametrize(
    ("shell_part", "expected_status"),
    [
        ({"type": "input_audio"}, 200),
        ({"type": "video_url"}, 200),
        ({"type": "file"}, 500),  # the provider side rejects a payload-less file part downstream
    ],
)
def test_bedrock_pre_call_drops_unscannable_shell_part(
    gateway: Gateway, tmp_path: Path, shell_part: dict[str, JsonValue], expected_status: int
) -> None:
    def guardrail(request: Request) -> Reply:
        assert request.target == "/guardrail/synthetic-guardrail/version/1/apply"
        body: Final = json.loads(request.body)
        assert body["content"] == [{"text": {"text": "hello"}}]
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    with wire_server(guardrail) as policy:
        path: Final = _guardrail_config(
            tmp_path, "bedrock-shell-" + uuid.uuid4().hex, _bedrock_policy(policy.url), "bedrock-shell.yaml"
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as observed:
                observed.get("/__observations")
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, shell_part]}],
                    },
                )
                assert response.status_code == expected_status, response.text
                assert "cannot be scanned" not in response.text, response.text
                if expected_status == 200:
                    assert len(observed.get("/__observations").json()["requests"]) == 1
            assert len(policy.drain()) == 1


def test_non_bedrock_policy_forwards_non_string_image_url(gateway: Gateway, tmp_path: Path) -> None:
    def sink(_request: Request) -> Reply:
        return Reply(body=b'{"action":"NONE"}')

    with wire_server(sink) as policy:
        params: Final = {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": True,
            "api_base": policy.url,
            "api_key": "synthetic-guardrail-key",
        }
        path: Final = _guardrail_config(tmp_path, "generic-img-" + uuid.uuid4().hex, params, "generic-img.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "look"},
                                {"type": "image_url", "image_url": {"url": 5}},
                            ],
                        }
                    ],
                },
            )
            assert response.status_code == 500, response.text
            assert "input_value=5" in response.text, response.text


def test_bedrock_during_call_tool_output_without_payload_keys_scans_as_text(gateway: Gateway, tmp_path: Path) -> None:
    tool_output: Final = json.dumps(
        [
            {"type": "file", "name": "README.md", "path": "README.md", "size": 1200},
            {"type": "dir", "name": "src", "path": "src"},
        ]
    )

    def guardrail(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        texts: Final = [item["text"]["text"] for item in body["content"] if "text" in item]
        assert tool_output in texts, body
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    def chat_reply(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-wire",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(chat_reply) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        bedrock_params: Final = _bedrock_policy(policy.url)
        bedrock_params["mode"] = "during_call"
        config["guardrails"] = [
            {"guardrail_name": "bedrock-during-" + uuid.uuid4().hex, "litellm_params": bedrock_params}
        ]
        path: Final = tmp_path / "bedrock-tool-json.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=upstream.url)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "list files"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "t1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
                            ],
                        },
                        {"role": "tool", "tool_call_id": "t1", "content": tool_output},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            eventually(lambda: policy.drain(), lambda values: len(values) == 1, seconds=10)
            assert len(upstream.drain()) == 1


def test_bedrock_during_call_tool_output_url_members_scan_as_text(gateway: Gateway, tmp_path: Path) -> None:
    tool_output: Final = json.dumps(
        [
            {
                "type": "file",
                "name": "README.md",
                "path": "README.md",
                "size": 1200,
                "url": "https://api.github.com/repos/o/r/contents/README.md",
            },
            {"type": "dir", "name": "src", "path": "src", "url": "https://api.github.com/repos/o/r/contents/src"},
        ]
    )

    def guardrail(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        texts: Final = [item["text"]["text"] for item in body["content"] if "text" in item]
        assert tool_output in texts, body
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    def chat_reply(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-wire",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(chat_reply) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        bedrock_params: Final = _bedrock_policy(policy.url)
        bedrock_params["mode"] = "during_call"
        config["guardrails"] = [
            {"guardrail_name": "bedrock-during-" + uuid.uuid4().hex, "litellm_params": bedrock_params}
        ]
        path: Final = tmp_path / "bedrock-tool-url.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=upstream.url)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "list files"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "t1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
                            ],
                        },
                        {"role": "tool", "tool_call_id": "t1", "content": tool_output},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            eventually(lambda: policy.drain(), lambda values: len(values) == 1, seconds=10)
            assert len(upstream.drain()) == 1


def test_bedrock_during_call_responses_function_call_output_string_scans_as_text(
    gateway: Gateway, tmp_path: Path
) -> None:
    tool_output: Final = json.dumps([{"type": "file", "name": "q3.pdf", "file_id": "file-abc"}])

    def guardrail(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        texts: Final = [item["text"]["text"] for item in body["content"] if "text" in item]
        assert tool_output in texts, body
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    def responses_reply(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "resp-wire",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg-wire",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "done", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(responses_reply) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        bedrock_params: Final = _bedrock_policy(policy.url)
        bedrock_params["mode"] = "during_call"
        config["guardrails"] = [
            {"guardrail_name": "bedrock-during-" + uuid.uuid4().hex, "litellm_params": bedrock_params}
        ]
        path: Final = tmp_path / "bedrock-resp-tool-string.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=upstream.url)
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "input": [
                        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
                        {"type": "function_call_output", "call_id": "c1", "output": tool_output},
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "summarize"}],
                        },
                    ],
                },
            )
            assert response.status_code == 200, response.text
            eventually(lambda: policy.drain(), lambda values: len(values) == 1, seconds=10)
            assert len(upstream.drain()) == 1


def test_bedrock_during_call_skip_tool_message_flag_drops_tool_result_text(gateway: Gateway, tmp_path: Path) -> None:
    tool_text: Final = "BLOCKME from tool"

    def guardrail(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        texts: Final = [item["text"]["text"] for item in body["content"] if "text" in item]
        assert texts == ["hello"], body
        return Reply(body=b'{"action":"NONE","outputs":[],"assessments":[]}')

    with wire_server(guardrail) as policy, wire_server(_anthropic_reply) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        bedrock_params: Final = _bedrock_policy(policy.url)
        bedrock_params["mode"] = "during_call"
        bedrock_params["skip_tool_message_in_guardrail"] = True
        config["guardrails"] = [
            {"guardrail_name": "bedrock-skip-tool-" + uuid.uuid4().hex, "litellm_params": bedrock_params}
        ]
        path: Final = tmp_path / "bedrock-skip-tool.yaml"
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
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "toolu_01A", "name": "lookup", "input": {}}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "toolu_01A", "content": tool_text}],
                        },
                    ],
                },
            )
            assert response.status_code == 200, response.text
            eventually(lambda: policy.drain(), lambda values: len(values) == 1, seconds=10)
            assert len(upstream.drain()) == 1
