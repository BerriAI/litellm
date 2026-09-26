import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
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
            import httpx

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


def test_panw_latest_role_message_only_scans_only_latest_turn_on_responses_input(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    latest: Final = "latest turn " + uuid.uuid4().hex
    history: Final = ({"role": "user", "content": "first turn"}, {"role": "assistant", "content": "first reply"})
    shapes: Final = {
        "plain": {"input": [*history, {"role": "user", "content": latest}]},
        "instructions": {"instructions": "answer briefly", "input": [*history, {"role": "user", "content": latest}]},
        "function_call_output": {
            "input": [
                *history,
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "tool result"},
                {"role": "user", "content": latest},
            ]
        },
        "reasoning": {
            "input": [
                *history,
                {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {"role": "user", "content": latest},
            ]
        },
        "tool_loop_after_latest": {
            "input": [
                *history,
                {"role": "user", "content": latest},
                {"type": "reasoning", "id": "rs_2", "content": [{"type": "reasoning_text", "text": "thinking"}]},
                {"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_2", "output": "tool result"},
            ]
        },
    }

    def scanner(request: Request) -> Reply:
        assert request.target == "/v1/scan/sync/request"
        body: Final = json.loads(request.body)
        return Reply(
            body=json.dumps(
                {
                    "action": "allow",
                    "category": "benign",
                    "profile_name": "synthetic-profile",
                    "report_id": "R" + body["tr_id"],
                    "scan_id": "S" + body["tr_id"],
                    "tr_id": body["tr_id"],
                    "prompt_detected": {"injection": False, "url_cats": False, "dlp": False},
                    "response_detected": {},
                }
            ).encode()
        )

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/responses"
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_" + identity,
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_" + identity,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(scanner) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "panw_prisma_airs",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-panw-key",
                    "profile_name": "synthetic-profile",
                    "experimental_use_latest_role_message_only": True,
                },
            }
        ]
        path: Final = tmp_path / "panw.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            for name, shape in shapes.items():
                response = candidate.request("POST", "/v1/responses", {"model": model, **shape})
                assert response.status_code == 200, response.text
                assert response.json()["output"][0]["content"][0]["text"] == "permitted response"
                scanned = [json.loads(scan.body)["contents"][0]["prompt"] for scan in policy.drain()]
                assert scanned == [latest], f"{name}: latest-only scanned {scanned}"
                assert json.loads(upstream.drain()[0].body)["input"] == shape["input"]


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
