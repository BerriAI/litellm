import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_CITATION: Final = {
    "type": "char_location",
    "cited_text": "x",
    "document_index": 0,
    "document_title": "doc",
    "start_char_index": 0,
    "end_char_index": 5,
}


def _anthropic_stream(text: str) -> tuple[bytes, ...]:
    events: Final = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me recall "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "the contact record."}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCkgIARACClEK"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "EroBCoYBREDACTED=="}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("content_block_start", {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 2, "delta": {"type": "citations_delta", "citation": _CITATION}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        ("content_block_start", {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": '{"q": "jane"}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 3}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 42}}),
        ("message_stop", {"type": "message_stop"}),
    )
    return tuple(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode() for kind, data in events)


def _events(body: bytes) -> tuple[Mapping[str, object], ...]:
    return tuple(
        json.loads(line[len("data: ") :]) for line in body.decode("utf-8").splitlines() if line.startswith("data: ")
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


@pytest.mark.covers("other.observability.guardrails.presidio_masks_anthropic_stream_without_rewriting_other_frames")
def test_presidio_output_masking_keeps_anthropic_stream_frames_intact(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "presidio" + uuid.uuid4().hex
    email: Final = "jane.doe@example.com"
    pii_text: Final = "Contact Jane Doe at jane.doe@example.com for details."
    clean_text: Final = "Contact the support team for details."

    def presidio(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        text: Final = body.get("text", "")
        if request.target.endswith("/analyze"):
            results: Final = [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.99,
                    "analysis_explanation": None,
                    "recognition_metadata": None,
                }
                for match in re.finditer(re.escape(email), text)
            ]
            return Reply(body=json.dumps(results).encode())
        if request.target.endswith("/anonymize"):
            masked: Final = text.replace(email, "<EMAIL_ADDRESS>")
            items: Final = [
                {
                    "operator": "replace",
                    "entity_type": "EMAIL_ADDRESS",
                    "start": match.start(),
                    "end": match.end(),
                    "text": "<EMAIL_ADDRESS>",
                }
                for match in re.finditer(re.escape("<EMAIL_ADDRESS>"), masked)
            ]
            return Reply(body=json.dumps({"text": masked, "items": items}).encode())
        return Reply(status=404)

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        sent: Final = json.loads(request.body)
        content: Final = sent["messages"][0]["content"]
        assert content in ("pii control", "clean control"), content
        frames: Final = _anthropic_stream(pii_text if content == "pii control" else clean_text)
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(presidio) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "presidio",
                    "mode": "post_call",
                    "default_on": True,
                    "apply_to_output": True,
                    "presidio_analyzer_api_base": f"{policy.url}/analyzer/",
                    "presidio_anonymizer_api_base": f"{policy.url}/anonymizer/",
                },
            }
        ]
        path: Final = tmp_path / "presidio.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=upstream.url, api_key="synthetic-anthropic-key"
            )

            def messages_request(markers: str) -> httpx.Response:
                return candidate.request(
                    "POST",
                    "/v1/messages",
                    {"model": model, "stream": True, "max_tokens": 256, "messages": [{"role": "user", "content": markers}]},
                )

            masked: Final = messages_request("pii control")
            assert masked.status_code == 200, masked.text
            body: Final = masked.content
            decoded: Final = body.decode("utf-8", errors="replace")
            events: Final = _events(body)
            assert (
                "".join(
                    event["delta"]["text"]
                    for event in events
                    if event["type"] == "content_block_delta"
                    and event["index"] == 2
                    and event["delta"].get("type") == "text_delta"
                )
                == "Contact Jane Doe at <EMAIL_ADDRESS> for details."
            ), decoded
            assert email.encode() not in body, decoded
            assert (
                "".join(
                    event["delta"]["thinking"]
                    for event in events
                    if event["type"] == "content_block_delta" and event["delta"].get("type") == "thinking_delta"
                )
                == "Let me recall the contact record."
            ), decoded
            redacted: Final = tuple(
                event["content_block"] for event in events if event["type"] == "content_block_start" and event["index"] == 1
            )
            assert redacted == ({"type": "redacted_thinking", "data": "EroBCoYBREDACTED=="},), decoded
            signatures: Final = tuple(
                event["delta"]["signature"]
                for event in events
                if event["type"] == "content_block_delta" and event["delta"].get("type") == "signature_delta"
            )
            assert signatures == ("EqQBCkgIARACClEK",), decoded
            citations: Final = tuple(
                event["delta"]["citation"]
                for event in events
                if event["type"] == "content_block_delta" and event["delta"].get("type") == "citations_delta"
            )
            assert citations == (_CITATION,), decoded
            tool_starts: Final = tuple(
                event["content_block"] for event in events if event["type"] == "content_block_start" and event["index"] == 3
            )
            assert tool_starts == ({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},), decoded
            json_parts: Final = tuple(
                event["delta"]["partial_json"]
                for event in events
                if event["type"] == "content_block_delta" and event["delta"].get("type") == "input_json_delta"
            )
            assert json_parts == ('{"q": "jane"}',), decoded
            kinds: Final = tuple(event["type"] for event in events)
            assert kinds.count("message_start") == 1 and kinds.count("message_delta") == 1 and kinds.count("message_stop") == 1, decoded
            assert kinds[0] == "message_start" and kinds[-2] == "message_delta" and kinds[-1] == "message_stop", decoded
            assert events[-2]["delta"]["stop_reason"] == "tool_use", decoded

            clean: Final = messages_request("clean control")
            assert clean.status_code == 200, clean.text

            def projected(body_bytes: bytes) -> tuple[dict, ...]:
                return tuple(
                    {
                        key: value
                        for key, value in event.items()
                        if key in ("type", "index", "delta", "content_block")
                    }
                    for event in _events(body_bytes)
                )

            clean_decoded: Final = clean.content.decode("utf-8", errors="replace")
            assert projected(clean.content) == projected(b"".join(_anthropic_stream(clean_text))), clean_decoded

            analyzed: Final = tuple(
                json.loads(item.body)["text"] for item in policy.drain() if item.target.endswith("/analyze")
            )
            assert pii_text in analyzed, analyzed
            assert not any("Let me recall" in text for text in analyzed), analyzed
