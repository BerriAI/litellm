import asyncio
import json
import os
import re
import signal
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.mcp import mcp_peer, register_mcp, tool_names
from integration._support.process import OwnedProxy, group_members, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue


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


def test_panw_scans_and_masks_top_level_instructions_on_responses_input(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    ssn: Final = "123-45-6789"
    instructions: Final = "Never repeat the SSN " + ssn + " back " + uuid.uuid4().hex
    latest: Final = "latest turn " + uuid.uuid4().hex
    shapes: Final = {
        "list_input": ([{"role": "user", "content": "first turn"}, {"role": "user", "content": latest}], "first turn"),
        "string_input": (latest, None),
    }

    def scanner(request: Request) -> Reply:
        assert request.target == "/v1/scan/sync/request"
        body: Final = json.loads(request.body)
        prompt: Final = body["contents"][0]["prompt"]
        masked: Final = {"prompt_masked_data": {"data": prompt.replace(ssn, "<US_SSN>")}} if ssn in prompt else {}
        return Reply(
            body=json.dumps(
                {
                    "action": "allow",
                    "category": "dlp" if masked else "benign",
                    "profile_name": "synthetic-profile",
                    "report_id": "R" + body["tr_id"],
                    "scan_id": "S" + body["tr_id"],
                    "tr_id": body["tr_id"],
                    "prompt_detected": {"injection": False, "url_cats": False, "dlp": bool(masked)},
                    "response_detected": {},
                    **masked,
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
                },
            }
        ]
        path: Final = tmp_path / "panw.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            for name, (shape, first_turn) in shapes.items():
                response = candidate.request(
                    "POST", "/v1/responses", {"model": model, "instructions": instructions, "input": shape}
                )
                assert response.status_code == 200, response.text
                assert response.json()["output"][0]["content"][0]["text"] == "permitted response"
                scanned = [json.loads(scan.body)["contents"][0]["prompt"] for scan in policy.drain()]
                expected = [instructions, *([first_turn] if first_turn else []), latest]
                assert scanned == expected, f"{name}: scanned {scanned}"
                sent = json.loads(upstream.drain()[0].body)
                assert sent["instructions"] == instructions.replace(ssn, "<US_SSN>"), f"{name}: sent {sent}"
                assert sent["input"] == shape, f"{name}: sent {sent}"


_SSN: Final = "123-45-6789"
_MASKED_SSN: Final = "<US_SSN>"
_DENIED_TERM: Final = "RIGBLOCKME"


def _panw_scanner(request: Request) -> Reply:
    assert request.target == "/v1/scan/sync/request"
    body: Final = json.loads(request.body)
    prompt: Final = body["contents"][0]["prompt"]
    denied: Final = _DENIED_TERM in prompt
    masked: Final = {"prompt_masked_data": {"data": prompt.replace(_SSN, _MASKED_SSN)}} if _SSN in prompt else {}
    return Reply(
        body=json.dumps(
            {
                "action": "block" if denied else "allow",
                "category": "malicious" if denied else ("dlp" if masked else "benign"),
                "profile_name": "synthetic-profile",
                "report_id": "R" + body["tr_id"],
                "scan_id": "S" + body["tr_id"],
                "tr_id": body["tr_id"],
                "prompt_detected": {"injection": denied, "url_cats": False, "dlp": bool(masked)},
                "response_detected": {},
                **masked,
            }
        ).encode()
    )


def _responses_provider(request: Request) -> Reply:
    if request.method == "GET" and request.target.endswith("/models"):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    assert request.target == "/v1/responses", request.target
    return Reply(
        body=json.dumps(
            {
                "id": "resp_" + uuid.uuid4().hex,
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "gpt-4.1-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_synthetic",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _panw_config(tmp_path: Path, identity: str, policy_url: str, **flags: bool) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "panw_prisma_airs",
                "mode": "pre_call",
                "default_on": True,
                "api_base": policy_url,
                "api_key": "synthetic-panw-key",
                "profile_name": "synthetic-profile",
                **flags,
            },
        }
    ]
    path: Final = tmp_path / "panw.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _scanned_prompts(scans: tuple[Request, ...]) -> list[str]:
    return [json.loads(scan.body)["contents"][0]["prompt"] for scan in scans]


def _forwarded_bodies(requests: tuple[Request, ...]) -> list[dict[str, object]]:
    return [json.loads(request.body) for request in requests if request.method == "POST"]


def test_guardrail_denies_responses_request_whose_only_flagged_text_is_in_instructions(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    instructions: Final = "You are terse and say " + _DENIED_TERM + " " + uuid.uuid4().hex
    shapes: Final = {"string_input": "say hi", "list_input": [{"role": "user", "content": "say hi"}]}
    with wire_server(_panw_scanner) as policy, wire_server(_responses_provider) as upstream:
        config: Final = _panw_config(tmp_path, identity, policy.url)
        with (
            owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            for name, shape in shapes.items():
                response = candidate.request(
                    "POST", "/v1/responses", {"model": model, "instructions": instructions, "input": shape}
                )
                assert response.status_code == 400, f"{name}: {response.text}"
                assert "Prompt blocked by PANW Prisma AI Security policy" in response.text, response.text
                assert _scanned_prompts(policy.drain()) == [instructions], name
                assert _forwarded_bodies(upstream.drain()) == [], (
                    f"{name}: denied instructions must not reach the provider"
                )


def test_empty_instructions_are_not_scanned_while_input_and_chat_system_masking_are_unchanged(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    secret: Final = "my SSN is " + _SSN + " " + uuid.uuid4().hex
    masked: Final = secret.replace(_SSN, _MASKED_SSN)

    def chat_provider(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions"
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl_" + identity,
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )

    def provider(request: Request) -> Reply:
        return chat_provider(request) if request.target == "/v1/chat/completions" else _responses_provider(request)

    with wire_server(_panw_scanner) as policy, wire_server(provider) as upstream:
        config: Final = _panw_config(tmp_path, identity, policy.url, mask_request_content=True)
        with (
            owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            for instructions in ("", None):
                body = {"model": model, "input": secret, **({} if instructions is None else {"instructions": ""})}
                response = candidate.request("POST", "/v1/responses", body)
                assert response.status_code == 200, response.text
                assert _scanned_prompts(policy.drain()) == [secret], f"instructions={instructions!r}"
                (sent,) = _forwarded_bodies(upstream.drain())
                assert sent.get("instructions") == instructions, f"instructions={instructions!r}: sent {sent}"
                assert sent["input"] == masked, f"instructions={instructions!r}: sent {sent}"

            response = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "system", "content": secret}, {"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200, response.text
            assert _scanned_prompts(policy.drain()) == [secret, "hi"]
            (sent_chat,) = _forwarded_bodies(upstream.drain())
            assert sent_chat["messages"] == [
                {"role": "system", "content": masked},
                {"role": "user", "content": "hi"},
            ]


def test_skip_system_message_leaves_instructions_and_system_items_unscanned_on_responses(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    instructions: Final = "Escalations go to " + _SSN + " " + uuid.uuid4().hex
    system_item: Final = "House rules: never share " + _SSN + " " + uuid.uuid4().hex
    developer_item: Final = "Developer note " + _SSN + " " + uuid.uuid4().hex
    latest: Final = "my contact is " + _SSN + " " + uuid.uuid4().hex

    with wire_server(_panw_scanner) as policy, wire_server(_responses_provider) as upstream:
        config: Final = _panw_config(
            tmp_path, identity, policy.url, mask_request_content=True, skip_system_message_in_guardrail=True
        )
        with (
            owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response = candidate.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "instructions": instructions,
                    "input": [
                        {"role": "system", "content": system_item},
                        {"role": "developer", "content": developer_item},
                        {"role": "user", "content": latest},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert _scanned_prompts(policy.drain()) == [developer_item, latest]
            (sent,) = _forwarded_bodies(upstream.drain())
            assert sent["instructions"] == instructions, f"sent {sent}"
            assert sent["input"] == [
                {"role": "system", "content": system_item},
                {"role": "developer", "content": developer_item.replace(_SSN, _MASKED_SSN)},
                {"role": "user", "content": latest.replace(_SSN, _MASKED_SSN)},
            ], f"sent {sent}"


def test_instructions_masking_lands_next_to_multimodal_and_tool_loop_input_items(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    instructions: Final = "Never repeat the SSN " + _SSN + " back " + uuid.uuid4().hex
    latest: Final = "latest turn with " + _SSN + " " + uuid.uuid4().hex
    image: Final = {"type": "input_image", "image_url": "https://example.test/receipt.png", "detail": "low"}
    shapes: Final = {
        "multimodal": [
            {"role": "user", "content": [{"type": "input_text", "text": "first turn"}, image]},
            {"role": "user", "content": [image, {"type": "input_text", "text": latest}]},
        ],
        "tool_loop": [
            {"role": "user", "content": "first turn"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "tool result with " + _SSN},
            {"role": "user", "content": latest},
        ],
    }
    with wire_server(_panw_scanner) as policy, wire_server(_responses_provider) as upstream:
        config: Final = _panw_config(tmp_path, identity, policy.url, mask_request_content=True)
        with (
            owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            for name, shape in shapes.items():
                response = candidate.request(
                    "POST", "/v1/responses", {"model": model, "instructions": instructions, "input": shape}
                )
                assert response.status_code == 200, f"{name}: {response.text}"
                assert _scanned_prompts(policy.drain()) == [instructions, "first turn", latest], name
                (sent,) = _forwarded_bodies(upstream.drain())
                assert sent["instructions"] == instructions.replace(_SSN, _MASKED_SSN), f"{name}: sent {sent}"
                expected = json.loads(json.dumps(shape).replace(latest, latest.replace(_SSN, _MASKED_SSN)))
                assert sent["input"] == expected, f"{name}: sent {sent}"


def test_panw_latest_only_with_instructions_masks_only_the_latest_turn(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    instructions: Final = "Keep " + _SSN + " confidential " + uuid.uuid4().hex
    latest: Final = "latest turn with " + _SSN + " " + uuid.uuid4().hex
    history: Final = ({"role": "user", "content": "first turn"}, {"role": "assistant", "content": "first reply"})
    shapes: Final = {
        "plain": [*history, {"role": "user", "content": latest}],
        "reasoning": [
            *history,
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"role": "user", "content": latest},
        ],
    }
    with wire_server(_panw_scanner) as policy, wire_server(_responses_provider) as upstream:
        config: Final = _panw_config(
            tmp_path, identity, policy.url, mask_request_content=True, experimental_use_latest_role_message_only=True
        )
        with (
            owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            for name, shape in shapes.items():
                response = candidate.request(
                    "POST", "/v1/responses", {"model": model, "instructions": instructions, "input": shape}
                )
                assert response.status_code == 200, f"{name}: {response.text}"
                assert _scanned_prompts(policy.drain()) == [latest], name
                (sent,) = _forwarded_bodies(upstream.drain())
                assert sent["instructions"] == instructions, f"{name}: latest-only must leave instructions alone"
                assert sent["input"] == [*shape[:-1], {"role": "user", "content": latest.replace(_SSN, _MASKED_SSN)}], (
                    f"{name}: sent {sent}"
                )


def test_bedrock_latest_only_masks_latest_turn_on_responses_input_with_instructions(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    instructions: Final = "Keep " + _SSN + " confidential " + uuid.uuid4().hex
    latest: Final = "latest turn with " + _SSN + " " + uuid.uuid4().hex

    def guardrail(request: Request) -> Reply:
        assert request.target == f"/guardrail/{guardrail_id}/version/DRAFT/apply", request.target
        body: Final = json.loads(request.body)
        assert body["source"] == "INPUT", body
        assert body["content"] == [{"text": {"text": latest}}], body
        return Reply(
            body=json.dumps(
                {
                    "action": "GUARDRAIL_INTERVENED",
                    "outputs": [{"text": latest.replace(_SSN, _MASKED_SSN)}],
                    "assessments": [
                        {
                            "sensitiveInformationPolicy": {
                                "piiEntities": [
                                    {"type": "US_SOCIAL_SECURITY_NUMBER", "match": _SSN, "action": "ANONYMIZED"}
                                ]
                            }
                        }
                    ],
                }
            ).encode()
        )

    with wire_server(guardrail) as policy, wire_server(_responses_provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "bedrock",
                    "mode": "pre_call",
                    "default_on": True,
                    "mask_request_content": True,
                    "experimental_use_latest_role_message_only": True,
                    "guardrailIdentifier": guardrail_id,
                    "guardrailVersion": "DRAFT",
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
                    "aws_secret_access_key": "synthetic-secret",
                    "aws_bedrock_runtime_endpoint": policy.url,
                },
            }
        ]
        path: Final = tmp_path / "bedrock-instructions.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {
                    "model": model,
                    "instructions": instructions,
                    "input": [
                        {"role": "user", "content": "first turn"},
                        {"role": "assistant", "content": "first reply"},
                        {"role": "user", "content": latest},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert len(policy.drain()) == 1
            (sent,) = _forwarded_bodies(upstream.drain())
            assert sent["instructions"] == instructions, sent
            assert sent["input"] == [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": latest.replace(_SSN, _MASKED_SSN)},
            ], sent


def test_instructions_masking_holds_under_concurrent_load_across_two_workers(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    with wire_server(_panw_scanner) as policy, wire_server(_responses_provider) as upstream:
        config: Final = _panw_config(tmp_path, identity, policy.url, mask_request_content=True)
        with (
            owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            tags: Final = tuple(uuid.uuid4().hex for _ in range(16))

            def send(tag: str) -> httpx.Response:
                return candidate.request(
                    "POST",
                    "/v1/responses",
                    {"model": model, "instructions": "Keep " + _SSN + " private " + tag, "input": "say hi " + tag},
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                responses: Final = tuple(pool.map(send, tags))
            assert [response.status_code for response in responses] == [200] * len(tags), [
                response.text for response in responses
            ]
            sent: Final = {str(body["input"]): body for body in _forwarded_bodies(upstream.drain())}
            assert sorted(_scanned_prompts(policy.drain())) == sorted(
                [text for tag in tags for text in ("Keep " + _SSN + " private " + tag, "say hi " + tag)]
            )
            assert {tag: sent["say hi " + tag]["instructions"] for tag in tags} == {
                tag: "Keep " + _MASKED_SSN + " private " + tag for tag in tags
            }


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
@pytest.mark.parametrize(
    ("logging_only_scope", "scanned_directions"),
    (("input", ("request",)), ("output", ("response",)), ("both", ("request", "response"))),
)
def test_logging_only_scope_observes_only_the_configured_direction_without_blocking(
    gateway: Gateway, tmp_path: Path, logging_only_scope: str, scanned_directions: tuple[str, ...]
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    prompt: Final = "synthetic observed prompt " + identity
    reply: Final = "synthetic observed reply " + identity
    texts_by_direction: Final = {"request": [prompt], "response": [reply]}

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic observed denial"}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions"
        assert json.loads(request.body)["messages"] == [{"role": "user", "content": prompt}]
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
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
                    "mode": "logging_only",
                    "logging_only_scope": logging_only_scope,
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "logging-only-scope.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=upstream.url + "/v1")
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}]}
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == reply, response.text
            assert len(upstream.drain()) == 1
            rows: Final = eventually(
                lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
                lambda values: len(values) == 1,
                seconds=70,
            )
            scans: Final = tuple(json.loads(scan.body) for scan in policy.drain())
            assert [(scan["input_type"], scan["texts"]) for scan in scans] == [
                (direction, texts_by_direction[direction]) for direction in scanned_directions
            ], scans
            entries: Final = object_value(rows[0]["metadata"])["guardrail_information"]
            assert isinstance(entries, list), rows[0]
            assert [
                (entry["guardrail_name"], entry["guardrail_mode"], entry["guardrail_status"])
                for entry in map(object_value, entries)
            ] == [(identity, "logging_only", "guardrail_intervened")] * len(scanned_directions), entries
            today: Final = datetime.now(timezone.utc).date().isoformat()
            guardrail_id: Final = next(
                object_value(row)["guardrail_id"]
                for row in candidate.get("/v2/guardrails/list")["guardrails"]
                if object_value(row)["guardrail_name"] == identity
            )
            detail: Final = eventually(
                lambda: candidate.request(
                    "GET",
                    f"/guardrails/usage/detail/{guardrail_id}",
                    params={"start_date": today, "end_date": today},
                ).json(),
                lambda body: body["requestsEvaluated"] >= len(scanned_directions),
                seconds=30,
                return_last_on_timeout=True,
            )
            assert detail["requestsEvaluated"] == len(scanned_directions), detail


@pytest.mark.parametrize("logging_only_scope", ("input", "Input"))
def test_logging_only_scope_literal_or_mode_mismatch_is_ignored_at_load_and_keeps_blocking(
    gateway: Gateway, tmp_path: Path, logging_only_scope: str
) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    prompt: Final = "synthetic invalid-scope prompt pineapple " + identity

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api"
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions"
        assert json.loads(request.body)["messages"] == [{"role": "user", "content": prompt}]
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "unchanged provider reply"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
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
                    "logging_only_scope": logging_only_scope,
                    "default_on": True,
                    "blocked_words": [{"keyword": "pineapple", "action": "BLOCK"}],
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ]
        path: Final = tmp_path / "invalid-scope-pre-call.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=upstream.url + "/v1")
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}]}
            )
            assert response.status_code == 400, response.text
            assert "synthetic policy denial" in response.text, response.text
            assert len(policy.drain()) == 1
            assert len(upstream.drain()) == 0
            guardrails: Final = candidate.get("/v2/guardrails/list")["guardrails"]
            assert any(object_value(row)["guardrail_name"] == identity for row in guardrails), guardrails
_TOKEN: Final = re.compile(rb"token-[0-9a-f]{32}-\d+")


def _secret_for(request: Request) -> str:
    token: Final = _TOKEN.search(request.body)
    assert token is not None, request.body
    return "synthetic-leaked-secret-" + token.group().decode()


def _chat_frame(identity: str, choices: tuple[dict[str, JsonValue], ...]) -> bytes:
    payload: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": list(choices),
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chat_choice(index: int, delta: dict[str, JsonValue], finish: str | None = None) -> dict[str, JsonValue]:
    return {"index": index, "delta": delta, "finish_reason": finish}


def _chat_stream_frames(secret: str, shape: str) -> tuple[bytes, ...]:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    tool_call: Final = {
        "index": 0,
        "id": "call_" + identity,
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"query": secret})},
    }
    released: Final = {
        "text": (_chat_choice(0, {"role": "assistant", "content": secret}),),
        "empty": (_chat_choice(0, {"role": "assistant", "content": ""}),),
        "tool_call": (_chat_choice(0, {"role": "assistant", "tool_calls": [tool_call]}),),
        "two_choices": (
            _chat_choice(0, {"role": "assistant", "content": secret + "-first"}),
            _chat_choice(1, {"role": "assistant", "content": secret + "-second"}),
        ),
    }[shape]
    finish: Final = "tool_calls" if shape == "tool_call" else "stop"
    tail: Final = tuple(_chat_choice(int(str(choice["index"])), {"content": " tail"}, finish) for choice in released)
    return (_chat_frame(identity, released), _chat_frame(identity, tail), b"data: [DONE]\n\n")


def _chat_completion_body(secret: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": secret}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        }
    ).encode()


def _responses_tool_call_frames(secret: str, *, with_text: bool) -> tuple[bytes, ...]:
    identity: Final = "resp_" + uuid.uuid4().hex
    arguments: Final = json.dumps({"query": secret})
    pending: Final = {"type": "function_call", "id": "fc_" + identity, "call_id": "call_" + identity, "name": "lookup"}
    finished: Final = {**pending, "arguments": arguments, "status": "completed"}
    envelope: Final = {"id": identity, "object": "response", "created_at": 1, "model": "gpt-4o-mini", "output": []}
    message: Final = {
        "type": "message",
        "id": "msg_" + identity,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": secret, "annotations": []}],
    }
    text_delta: Final = {
        "type": "response.output_text.delta",
        "item_id": "msg_" + identity,
        "output_index": 0,
        "content_index": 0,
        "delta": secret,
    }
    text_events: Final = (text_delta,) if with_text else ()
    tool_index: Final = len(text_events)
    output: Final = [*((message,) if with_text else ()), finished]
    events: Final = (
        {"type": "response.created", "response": {**envelope, "status": "in_progress"}},
        *text_events,
        {
            "type": "response.output_item.added",
            "output_index": tool_index,
            "item": {**pending, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_" + identity,
            "output_index": tool_index,
            "delta": arguments,
        },
        {"type": "response.output_item.done", "output_index": tool_index, "item": finished},
        {"type": "response.completed", "response": {**envelope, "status": "completed", "output": output}},
    )
    encoded: Final = tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
    released_through: Final = len(events) - 1 if with_text else 3
    return (b"".join(encoded[:released_through]), b"".join(encoded[released_through:]))


def _responses_stream_frames(secret: str) -> tuple[bytes, ...]:
    identity: Final = "resp_" + uuid.uuid4().hex
    completed: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + identity,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": secret, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    events: Final = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": [], "usage": None}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": secret,
        },
        {"type": "response.completed", "response": completed},
    )
    encoded: Final = tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
    return (encoded[0] + encoded[1], encoded[2])


def _gemini_stream_frames(secret: str) -> tuple[bytes, ...]:
    def frame(text: str, finish: str | None) -> bytes:
        candidate: Final = {
            "content": {"parts": [{"text": text}], "role": "model"},
            "index": 0,
            **({"finishReason": finish} if finish else {}),
        }
        payload: Final = {
            "candidates": [candidate],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "modelVersion": "gemini-2.5-flash",
        }
        return b"data: " + json.dumps(payload).encode() + b"\r\n\r\n"

    return (frame(secret, None), frame(" tail", "STOP"))


def _scripted_provider(gate: threading.Event | None, pause: float, shape: str) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        secret: Final = _secret_for(request)
        path: Final = request.target.split("?")[0]
        if path.endswith("/chat/completions") and not json.loads(request.body).get("stream"):
            return Reply(body=_chat_completion_body(secret))
        frames: Final = (
            _gemini_stream_frames(secret)
            if "streamGenerateContent" in path
            else (
                _responses_tool_call_frames(secret, with_text=shape == "text_then_tool_call")
                if shape in ("tool_call", "text_then_tool_call")
                else _responses_stream_frames(secret)
            )
            if path.endswith("/responses")
            else _chat_stream_frames(secret, shape)
        )
        return Reply(content_type="text/event-stream", chunks=frames, gate_after_first=gate, pause_between_chunks=pause)

    return provider


def _allowing_guardrail(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api", request.target
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _failing_response_scans(reply: Reply) -> Callable[[Request], Reply]:
    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        if json.loads(request.body)["input_type"] == "response":
            return reply
        return Reply(body=json.dumps({"action": "NONE"}).encode())

    return guardrail


def _detect_only_pipeline_policy(identity: str) -> dict[str, JsonValue]:
    return {
        "guardrails": {"add": [identity]},
        "pipeline": {"mode": "post_call", "steps": [{"guardrail": identity, "on_pass": "allow", "on_fail": "next"}]},
    }


def _post_call_config(
    tmp_path: Path,
    identity: str,
    policy_url: str,
    params: Mapping[str, JsonValue],
    default_on: bool,
    *,
    pipeline: bool = False,
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "generic_guardrail_api",
                "mode": "post_call",
                "default_on": default_on,
                "api_base": policy_url,
                "api_key": "synthetic-guardrail-key",
                **params,
            },
        }
    ]
    if pipeline:
        config["policies"] = {f"{identity}-pipeline": _detect_only_pipeline_policy(identity)}
        config["policy_attachments"] = [{"policy": f"{identity}-pipeline", "scope": "*"}]
    path: Final = tmp_path / f"{identity}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class _ScanLog:
    def __init__(self, policy: Wire) -> None:
        self.policy: Final = policy
        self.seen: tuple[dict[str, JsonValue], ...] = ()

    def response_scans(self, secret: str) -> tuple[dict[str, JsonValue], ...]:
        self.seen = (*self.seen, *(object_value(json.loads(request.body)) for request in self.policy.drain()))
        return tuple(body for body in self.seen if body["input_type"] == "response" and secret in json.dumps(body))


@dataclass(frozen=True, slots=True)
class _DisconnectRig:
    owned: OwnedProxy
    model: str
    gemini: str
    scans: _ScanLog
    identity: str
    gate: threading.Event
    upstream: Wire

    @property
    def candidate(self) -> Gateway:
        return self.owned.gateway

    def token(self, index: int = 0) -> str:
        return f"token-{self.identity.removeprefix('guardrail')}-{index}"

    def secret(self, index: int = 0) -> str:
        return "synthetic-leaked-secret-" + self.token(index)


_END_OF_STREAM_ONLY: Final = MappingProxyType({"streaming_end_of_stream_only": True})


@contextmanager
def _disconnect_rig(
    gateway: Gateway,
    tmp_path: Path,
    *,
    params: Mapping[str, JsonValue] = _END_OF_STREAM_ONLY,
    guardrail: Callable[[Request], Reply] = _allowing_guardrail,
    gated: bool = True,
    pause: float = 0,
    shape: str = "text",
    default_on: bool = True,
    workers: int = 1,
    pipeline: bool = False,
) -> Iterator[_DisconnectRig]:
    identity: Final = "guardrail" + uuid.uuid4().hex
    gate: Final = threading.Event()
    with (
        wire_server(guardrail) as policy,
        wire_server(_scripted_provider(gate if gated else None, pause, shape)) as upstream,
    ):
        config: Final = _post_call_config(tmp_path, identity, policy.url, params, default_on, pipeline=pipeline)
        try:
            with (
                owned_proxy_process(gateway, tmp_path, {}, config=config, workers=workers) as owned,
                owned.gateway.scenario() as scenario,
            ):
                yield _DisconnectRig(
                    owned,
                    scenario.model(api_base=upstream.url + "/v1", api_key="synthetic-openai-key"),
                    scenario.model(
                        model="gemini/gemini-2.5-flash", api_base=upstream.url, api_key="synthetic-gemini-key"
                    ),
                    _ScanLog(policy),
                    identity,
                    gate,
                    upstream,
                )
        finally:
            gate.set()


def _close_on(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.read()
    for line in response.iter_lines():
        if marker in line:
            return line
    raise AssertionError(f"The stream ended before the client received {marker}")


def _stream_and_close(rig: _DisconnectRig, path: str, body: Mapping[str, JsonValue], marker: str) -> str:
    with rig.candidate.client.stream(
        "POST", path, json=dict(body), headers={"Authorization": f"Bearer {rig.candidate.key}"}
    ) as response:
        return _close_on(response, marker)


def _chat_body(rig: _DisconnectRig, index: int, stream: bool = True) -> dict[str, JsonValue]:
    return {
        "model": rig.model,
        "messages": [{"role": "user", "content": "synthetic prompt " + rig.token(index)}],
        "stream": stream,
    }


def _chat_httpx(rig: _DisconnectRig, index: int = 0) -> str:
    return _stream_and_close(rig, "/v1/chat/completions", _chat_body(rig, index), rig.secret(index))


def _responses_httpx(rig: _DisconnectRig, index: int = 0) -> str:
    body: Final = {"model": rig.model, "input": "synthetic prompt " + rig.token(index), "stream": True}
    return _stream_and_close(rig, "/v1/responses", body, rig.secret(index))


def _messages_httpx(rig: _DisconnectRig, index: int = 0) -> str:
    body: Final = {**_chat_body(rig, index), "max_tokens": 64}
    return _stream_and_close(rig, "/v1/messages", body, rig.secret(index))


def _gemini_httpx(rig: _DisconnectRig, index: int = 0) -> str:
    body: Final = {"contents": [{"role": "user", "parts": [{"text": "synthetic prompt " + rig.token(index)}]}]}
    path: Final = f"/v1beta/models/{rig.gemini}:streamGenerateContent?alt=sse"
    return _stream_and_close(rig, path, body, rig.secret(index))


def _chat_async_openai_sdk(rig: _DisconnectRig, index: int = 0) -> str:
    async def read() -> str:
        client: Final = AsyncOpenAI(
            base_url=str(rig.candidate.client.base_url) + "/v1",
            api_key=rig.candidate.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        )
        async with client:
            stream: Final = await client.chat.completions.create(
                model=rig.model,
                messages=[{"role": "user", "content": "synthetic prompt " + rig.token(index)}],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and rig.secret(index) in (chunk.choices[0].delta.content or ""):
                    await stream.close()
                    return chunk.choices[0].delta.content or ""
        raise AssertionError("The stream ended before the client received the streamed content")

    return asyncio.run(read())


def _responses_openai_sdk(rig: _DisconnectRig, index: int = 0) -> str:
    client: Final = OpenAI(
        base_url=str(rig.candidate.client.base_url) + "/v1",
        api_key=rig.candidate.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=30),
    )
    with client:
        stream: Final = client.responses.create(
            model=rig.model, input="synthetic prompt " + rig.token(index), stream=True
        )
        for event in stream:
            if event.type == "response.output_text.delta" and rig.secret(index) in event.delta:
                stream.close()
                return event.delta
    raise AssertionError("The stream ended before the client received the streamed content")


def _messages_anthropic_sdk(rig: _DisconnectRig, index: int = 0) -> str:
    client: Final = anthropic.Anthropic(
        base_url=str(rig.candidate.client.base_url),
        api_key=rig.candidate.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=30),
    )
    with client:
        stream: Final = client.messages.create(
            model=rig.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "synthetic prompt " + rig.token(index)}],
            stream=True,
        )
        for event in stream:
            text: Final = (
                event.delta.text if event.type == "content_block_delta" and event.delta.type == "text_delta" else ""
            )
            if rig.secret(index) in text:
                stream.close()
                return text
    raise AssertionError("The stream ended before the client received the streamed content")


def _scanned_while_upstream_is_held(
    rig: _DisconnectRig, disconnect: Callable[[_DisconnectRig, int], str], index: int = 0
) -> tuple[dict[str, JsonValue], ...]:
    try:
        received: Final = disconnect(rig, index)
        assert rig.secret(index) in received, received
        return eventually(
            lambda: rig.scans.response_scans(rig.secret(index)), lambda values: len(values) >= 1, seconds=4
        )
    finally:
        rig.gate.set()


def _post_call_statuses(rig: _DisconnectRig, model: str, rows: int = 1) -> tuple[tuple[str, ...], ...]:
    found: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda values: len(values) == rows,
        seconds=70,
    )

    def statuses(metadata: JsonValue) -> tuple[str, ...]:
        entries: Final = object_value(metadata).get("guardrail_information") or []
        assert isinstance(entries, list), metadata
        post_call: Final = tuple(
            entry
            for entry in (object_value(value) for value in entries)
            if entry.get("guardrail_name") == rig.identity and entry.get("guardrail_mode") == "post_call"
        )
        return tuple(str(entry["guardrail_status"]) for entry in post_call)

    return tuple(statuses(row["metadata"]) for row in found)


_ENDPOINT_CLIENTS: Final = (
    pytest.param(_chat_httpx, id="chat-httpx"),
    pytest.param(_chat_async_openai_sdk, id="chat-async-openai-sdk"),
    pytest.param(_responses_httpx, id="responses-httpx"),
    pytest.param(_responses_openai_sdk, id="responses-openai-sdk"),
    pytest.param(_messages_httpx, id="messages-httpx"),
    pytest.param(_messages_anthropic_sdk, id="messages-anthropic-sdk"),
    pytest.param(_gemini_httpx, id="native-gemini-stream-generate-content"),
)


_NO_DISCONNECT_ROW_LIT_8603: Final = pytest.mark.skip(
    reason="BUG: LIT-8603 a mid-stream disconnect writes no spend row"
)


@pytest.mark.parametrize("disconnect", _ENDPOINT_CLIENTS)
def test_client_disconnect_mid_stream_still_scans_the_content_it_already_received(
    gateway: Gateway, tmp_path: Path, disconnect: Callable[[_DisconnectRig, int], str]
) -> None:
    with _disconnect_rig(gateway, tmp_path) as rig:
        scans: Final = _scanned_while_upstream_is_held(rig, disconnect)
        assert len(scans) == 1, scans


@pytest.mark.parametrize(
    "disconnect",
    (
        pytest.param(_chat_httpx, id="chat-httpx"),
        pytest.param(_chat_async_openai_sdk, id="chat-async-openai-sdk"),
        pytest.param(_responses_httpx, id="responses-httpx", marks=_NO_DISCONNECT_ROW_LIT_8603),
        pytest.param(_messages_anthropic_sdk, id="messages-anthropic-sdk", marks=_NO_DISCONNECT_ROW_LIT_8603),
        pytest.param(
            _gemini_httpx,
            id="native-gemini-stream-generate-content",
            marks=pytest.mark.skip(reason="BUG: LIT-9087 a mid-stream disconnect writes no spend row"),
        ),
    ),
)
def test_client_disconnect_mid_stream_records_the_post_call_verdict_on_the_spend_row(
    gateway: Gateway, tmp_path: Path, disconnect: Callable[[_DisconnectRig, int], str]
) -> None:
    with _disconnect_rig(gateway, tmp_path) as rig:
        _scanned_while_upstream_is_held(rig, disconnect)
        model: Final = rig.gemini if disconnect is _gemini_httpx else rig.model
        assert _post_call_statuses(rig, model) == (("success",),)


@pytest.mark.parametrize(
    "reply",
    (
        pytest.param(Reply(status=500, body=b'{"error": "synthetic guardrail outage"}'), id="guardrail-500"),
        pytest.param(Reply(body=b"synthetic non-json guardrail body"), id="guardrail-malformed-200"),
    ),
)
def test_client_disconnect_mid_stream_records_a_failed_scan_when_the_guardrail_errors(
    gateway: Gateway, tmp_path: Path, reply: Reply
) -> None:
    with _disconnect_rig(gateway, tmp_path, guardrail=_failing_response_scans(reply)) as rig:
        _scanned_while_upstream_is_held(rig, _chat_httpx)
        assert _post_call_statuses(rig, rig.model) == (("guardrail_failed_to_respond",),)


def test_client_disconnect_mid_stream_records_a_blocking_verdict_and_keeps_serving(
    gateway: Gateway, tmp_path: Path
) -> None:
    blocked: Final = Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic leak"}).encode())
    with _disconnect_rig(gateway, tmp_path, guardrail=_failing_response_scans(blocked)) as rig:
        _scanned_while_upstream_is_held(rig, _chat_httpx)
        statuses: Final = _post_call_statuses(rig, rig.model)
        assert len(statuses) == 1 and len(statuses[0]) == 1 and statuses[0][0] != "success", statuses
        health: Final = rig.candidate.request("GET", "/health/liveliness")
        assert health.status_code == 200, health.text


def test_client_disconnect_while_end_of_stream_scan_is_in_flight_still_records_the_verdict(
    gateway: Gateway, tmp_path: Path
) -> None:
    scan_started: Final = threading.Event()
    scan_released: Final = threading.Event()

    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        if json.loads(request.body)["input_type"] == "response":
            scan_started.set()
            assert scan_released.wait(timeout=30), "The in-flight scan was never released"
        return Reply(body=json.dumps({"action": "NONE"}).encode())

    with _disconnect_rig(gateway, tmp_path, guardrail=guardrail, gated=False) as rig:
        with rig.candidate.client.stream(
            "POST",
            "/v1/chat/completions",
            json=_chat_body(rig, 0),
            headers={"Authorization": f"Bearer {rig.candidate.key}"},
        ) as response:
            try:
                assert rig.secret() in _close_on(response, rig.secret())
                assert scan_started.wait(timeout=10), "The end-of-stream scan never started"
            finally:
                pass
        scan_released.set()
        assert _post_call_statuses(rig, rig.model) == (("success",),)
        assert len(rig.scans.response_scans(rig.secret())) == 1


@pytest.mark.parametrize(
    ("params", "shape", "expected"),
    (
        pytest.param(
            {"streaming_buffer_until_moderated": False},
            "text",
            ("synthetic-leaked-secret-",),
            id="sampled-before-the-sampling-threshold",
        ),
        pytest.param(
            {"streaming_buffer_until_moderated": False, "streaming_transform_mode": "incremental_diff"},
            "tool_call",
            ('\\"query\\": \\"synthetic-leaked-secret-',),
            id="incremental-diff-tool-call-in-flight",
        ),
        pytest.param(dict(_END_OF_STREAM_ONLY), "two_choices", ("-first", "-second"), id="two-choices"),
    ),
)
def test_client_disconnect_mid_stream_scans_what_each_streaming_mode_released(
    gateway: Gateway, tmp_path: Path, params: dict[str, JsonValue], shape: str, expected: tuple[str, ...]
) -> None:
    with _disconnect_rig(gateway, tmp_path, params=params, shape=shape) as rig:
        marker: Final = rig.secret() + ("-first" if shape == "two_choices" else "")
        try:
            received: Final = _stream_and_close(rig, "/v1/chat/completions", _chat_body(rig, 0), rig.token())
            assert rig.token() in received, received
            scans: Final = eventually(
                lambda: rig.scans.response_scans(rig.secret()), lambda values: len(values) >= 1, seconds=4
            )
        finally:
            rig.gate.set()
        payload: Final = json.dumps(scans[-1])
        assert all(fragment in payload for fragment in expected), (marker, scans)
        assert _post_call_statuses(rig, rig.model)[0][-1:] == ("success",)


def _tool_call_request(rig: _DisconnectRig, path: str) -> dict[str, JsonValue]:
    if path == "/v1/responses":
        return {"model": rig.model, "input": "synthetic prompt " + rig.token(), "stream": True}
    if path == "/v1/messages":
        return {**_chat_body(rig, 0), "max_tokens": 64}
    return _chat_body(rig, 0)


@pytest.mark.parametrize(
    "path",
    (
        pytest.param("/v1/chat/completions", id="chat"),
        pytest.param("/v1/responses", id="responses"),
        pytest.param("/v1/messages", id="messages"),
    ),
)
def test_client_disconnect_mid_tool_call_scans_the_tool_call_it_already_received(
    gateway: Gateway, tmp_path: Path, path: str
) -> None:
    with _disconnect_rig(gateway, tmp_path, shape="tool_call") as rig:
        try:
            received: Final = _stream_and_close(rig, path, _tool_call_request(rig, path), rig.token())
            assert rig.token() in received, received
            scans: Final = eventually(
                lambda: rig.scans.response_scans(rig.secret()), lambda values: len(values) >= 1, seconds=4
            )
        finally:
            rig.gate.set()
        assert rig.secret() in json.dumps(scans[-1].get("tool_calls")), scans


def test_client_disconnect_after_a_finished_responses_tool_call_scans_the_text_and_tool_call_it_received(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _disconnect_rig(gateway, tmp_path, shape="text_then_tool_call") as rig:
        try:
            received: Final = _stream_and_close(
                rig, "/v1/responses", _tool_call_request(rig, "/v1/responses"), "response.output_item.done"
            )
            assert rig.token() in received, received
            scans: Final = eventually(
                lambda: tuple(scan for scan in rig.scans.response_scans(rig.secret()) if scan.get("texts")),
                lambda values: len(values) >= 1,
                seconds=4,
            )
        finally:
            rig.gate.set()
        assert rig.secret() in json.dumps(scans[-1].get("texts")), scans
        assert rig.secret() in json.dumps(scans[-1].get("tool_calls")), scans


def test_client_disconnect_mid_stream_scans_for_a_guardrail_the_request_opted_into(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _disconnect_rig(gateway, tmp_path, default_on=False) as rig:
        body: Final = {**_chat_body(rig, 0), "guardrails": [rig.identity]}
        try:
            received: Final = _stream_and_close(rig, "/v1/chat/completions", body, rig.secret())
            assert rig.secret() in received, received
            eventually(lambda: rig.scans.response_scans(rig.secret()), lambda values: len(values) == 1, seconds=4)
        finally:
            rig.gate.set()
        assert _post_call_statuses(rig, rig.model) == (("success",),)


_LIVE_DETECT_ONLY_PIPELINE: Final = MappingProxyType({"streaming_buffer_until_moderated": False})


@pytest.mark.parametrize(
    "disconnect",
    (
        pytest.param(_chat_httpx, id="chat-httpx"),
        pytest.param(_responses_httpx, id="responses-httpx"),
        pytest.param(_messages_httpx, id="messages-httpx"),
    ),
)
def test_live_detect_only_pipeline_releases_chunks_before_the_scan_and_scans_what_a_disconnected_client_received(
    gateway: Gateway, tmp_path: Path, disconnect: Callable[[_DisconnectRig, int], str]
) -> None:
    with _disconnect_rig(
        gateway, tmp_path, params=_LIVE_DETECT_ONLY_PIPELINE, default_on=False, pipeline=True
    ) as rig:
        scans: Final = _scanned_while_upstream_is_held(rig, disconnect)
        assert len(scans) == 1, scans


def test_live_detect_only_pipeline_records_the_verdict_of_a_disconnected_stream_on_the_spend_row(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _disconnect_rig(
        gateway, tmp_path, params=_LIVE_DETECT_ONLY_PIPELINE, default_on=False, pipeline=True
    ) as rig:
        _scanned_while_upstream_is_held(rig, _chat_httpx)
        assert _post_call_statuses(rig, rig.model) == (("success",),)


def test_client_disconnect_before_any_content_sends_no_response_scan(gateway: Gateway, tmp_path: Path) -> None:
    with _disconnect_rig(gateway, tmp_path, shape="empty") as rig:
        try:
            _stream_and_close(rig, "/v1/chat/completions", _chat_body(rig, 0), "data: ")
        finally:
            rig.gate.set()
        rows: Final = _post_call_statuses(rig, rig.model)
        assert rig.scans.response_scans(rig.secret()) == (), rig.scans.seen
        assert len(rows) == 1 and "success" not in rows[0], rows


@pytest.mark.parametrize(
    ("params", "pipeline"),
    (
        pytest.param(dict(_END_OF_STREAM_ONLY), False, id="end-of-stream-only"),
        pytest.param({"streaming_buffer_until_moderated": True}, False, id="buffered"),
        pytest.param(dict(_LIVE_DETECT_ONLY_PIPELINE), True, id="live-detect-only-pipeline"),
    ),
)
def test_a_fully_read_stream_is_scanned_exactly_once(
    gateway: Gateway, tmp_path: Path, params: dict[str, JsonValue], pipeline: bool
) -> None:
    with _disconnect_rig(gateway, tmp_path, params=params, gated=False, default_on=not pipeline, pipeline=pipeline) as rig:
        response: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig, 0))
        assert response.status_code == 200, response.text
        assert rig.secret() in response.text and "[DONE]" in response.text, response.text
        assert _post_call_statuses(rig, rig.model) == (("success",),)
        assert len(rig.scans.response_scans(rig.secret())) == 1, rig.scans.seen


def _cached_twin_rows(rig: _DisconnectRig) -> tuple[tuple[str, ...], ...]:
    first: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig, 0, stream=False))
    second: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig, 0, stream=False))
    assert first.status_code == second.status_code == 200, (first.text, second.text)
    assert first.json()["choices"] == second.json()["choices"], (first.text, second.text)
    assert len(rig.upstream.drain()) == 1, "the second request must be served from the cache"
    return _post_call_statuses(rig, rig.model, rows=2)


def test_a_non_streaming_response_and_its_cache_hit_are_each_scanned_once(gateway: Gateway, tmp_path: Path) -> None:
    with _disconnect_rig(gateway, tmp_path, gated=False) as rig:
        rows: Final = _cached_twin_rows(rig)
        assert rows[0] == ("success",), rows
        assert len(rig.scans.response_scans(rig.secret())) == 2, (rows, rig.scans.seen)


def test_a_cache_hit_row_records_the_post_call_verdict_of_its_scan(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: LIT-9088 the cache-hit spend row drops the post_call verdict of the scan that ran on it")
    with _disconnect_rig(gateway, tmp_path, gated=False) as rig:
        assert _cached_twin_rows(rig) == (("success",), ("success",))


def test_concurrent_disconnects_during_a_guardrail_outage_each_record_exactly_one_verdict(
    gateway: Gateway, tmp_path: Path
) -> None:
    def guardrail(request: Request) -> Reply:
        assert request.target == "/beta/litellm_basic_guardrail_api", request.target
        body: Final = json.loads(request.body)
        index: Final = int(_secret_for(request).rsplit("-", 1)[1])
        if body["input_type"] == "response" and index % 3 == 0:
            return Reply(status=503, body=b'{"error": "synthetic guardrail outage"}')
        return Reply(body=json.dumps({"action": "NONE"}).encode())

    clients: Final = (_chat_httpx, _responses_httpx, _messages_httpx)
    with _disconnect_rig(gateway, tmp_path, guardrail=guardrail, gated=False, pause=3, workers=2) as rig:
        with ThreadPoolExecutor(max_workers=30) as pool:
            received: Final = tuple(pool.map(lambda index: clients[index % 3](rig, index), range(30)))
        assert all(rig.secret(index) in line for index, line in enumerate(received)), received
        scanned: Final = eventually(
            lambda: tuple(len(rig.scans.response_scans(rig.secret(index) + '"')) for index in range(30)),
            lambda counts: all(count >= 1 for count in counts),
            seconds=20,
        )
        assert scanned == (1,) * 30, scanned
        chat_rows: Final = _post_call_statuses(rig, rig.model, rows=10)
        assert sorted(chat_rows) == sorted(
            ("guardrail_failed_to_respond",) if index % 3 == 0 else ("success",) for index in range(0, 30, 3)
        ), chat_rows


def test_disconnect_scans_keep_recording_after_a_worker_is_killed(gateway: Gateway, tmp_path: Path) -> None:
    with _disconnect_rig(gateway, tmp_path, gated=False, pause=3, workers=2) as rig:
        members: Final = tuple(
            member for member in group_members(rig.owned.process.pid) if member.pid != rig.owned.process.pid
        )
        workers: Final = tuple(member for member in members if any("spawn_main" in part for part in member.cmdline()))
        assert len(workers) >= 2, members
        workers[0].send_signal(signal.SIGKILL)
        psutil.wait_procs((workers[0],), timeout=10)
        with ThreadPoolExecutor(max_workers=8) as pool:
            received: Final = tuple(pool.map(lambda index: _chat_httpx(rig, index), range(8)))
        assert all(rig.secret(index) in line for index, line in enumerate(received)), received
        assert _post_call_statuses(rig, rig.model, rows=8) == (("success",),) * 8
        assert rig.owned.process.poll() is None
