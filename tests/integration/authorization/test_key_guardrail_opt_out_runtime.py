import asyncio
import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from anthropic import Anthropic
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


def _denying_guardrail(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api"
    return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}).encode())


def _guardrail_config(policy_url: str, path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "guardrail" + uuid.uuid4().hex,
            "litellm_params": {
                "guardrail": "generic_guardrail_api",
                "mode": "pre_call",
                "default_on": True,
                "api_base": policy_url,
                "api_key": "synthetic-guardrail-key",
            },
        }
    ]
    path.write_text(yaml.safe_dump(config))
    return path


def _stored_metadata(token: str) -> dict[str, object]:
    rows: Final = read_rows(
        'SELECT metadata FROM "LiteLLM_VerificationToken" WHERE token = %s', (sha256(token.encode()).hexdigest(),)
    )
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _upstream_observations(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(timeout=5, trust_env=False) as client:
        drained: Final = object_value(client.get(f"{gateway.upstream_url}/__observations").json())
    requests: Final = drained["requests"]
    assert isinstance(requests, list)
    return tuple(object_value(entry) for entry in requests)


def _upstream_hits(gateway: Gateway, marker: str) -> int:
    return sum(1 for entry in _upstream_observations(gateway) if marker in json.dumps(entry.get("body")))


def _wire_hits(wire: Wire, marker: str) -> int:
    return sum(1 for request in wire.drain() if marker.encode() in request.body)


def _sink_hits(policy: Wire, marker: str) -> int:
    return sum(1 for request in policy.drain() if marker.encode() in request.body)


def _anthropic_provider(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages", request.target
    body: Final = json.loads(request.body)
    if body.get("stream") is True:
        identity: Final = "msg_" + uuid.uuid4().hex
        frames: Final = (
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "synthetic"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
            {"type": "message_stop"},
        )
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n".encode() for frame in frames),
        )
    return Reply(
        body=json.dumps(
            {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "synthetic"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        ).encode()
    )


def _chat(candidate: Gateway, model: str, key: str, marker: str, *, stream: bool) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream},
        key=key,
    )


def _messages(candidate: Gateway, model: str, key: str, marker: str, *, stream: bool) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/messages",
        {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "max_tokens": 16,
            "stream": stream,
        },
        key=key,
    )


def _responses(candidate: Gateway, model: str, key: str, marker: str, *, stream: bool) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/responses",
        {"model": model, "input": marker, "stream": stream},
        key=key,
    )


@pytest.mark.covers(
    "mgmt.key.disable_global_guardrails.runtime.denied_on_all_surfaces",
)
def test_guardrail_denies_non_exempt_key_on_all_surfaces(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_denying_guardrail) as policy, wire_server(_anthropic_provider) as anthropic_wire:
        config: Final = _guardrail_config(policy.url, tmp_path / "denied.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            openai_model: Final = scenario.model()
            claude_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929",
                api_base=anthropic_wire.url,
                api_key="synthetic-anthropic-key",
            )
            deepseek_model: Final = scenario.model(model="deepseek/gpt-4o-mini", api_base=gateway.upstream_url + "/v1")
            key: Final = scenario.key(models=[openai_model, claude_model, deepseek_model])
            surfaces: Final = (
                ("chat", openai_model, _chat),
                ("messages", claude_model, _messages),
                ("responses", deepseek_model, _responses),
            )
            for surface, model, call in surfaces:
                for stream in (False, True):
                    marker: Final = f"denied-{surface}-{stream}-{uuid.uuid4().hex}"
                    response: Final = call(candidate, model, key, marker, stream=stream)
                    response.read()
                    assert response.status_code == 400, f"{surface} stream={stream}: {response.text}"
                    assert "synthetic policy denial" in response.text, response.text
                    assert _sink_hits(policy, marker) == 1
                    assert _upstream_hits(gateway, marker) == 0
                    assert _wire_hits(anthropic_wire, marker) == 0


@pytest.mark.covers(
    "mgmt.key.disable_global_guardrails.runtime.exempt_on_all_surfaces_and_clients",
)
def test_guardrail_skipped_for_admin_exempt_key_on_all_surfaces_and_clients(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_denying_guardrail) as policy, wire_server(_anthropic_provider) as anthropic_wire:
        config: Final = _guardrail_config(policy.url, tmp_path / "exempt.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            openai_model: Final = scenario.model()
            claude_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929",
                api_base=anthropic_wire.url,
                api_key="synthetic-anthropic-key",
            )
            deepseek_model: Final = scenario.model(model="deepseek/gpt-4o-mini", api_base=gateway.upstream_url + "/v1")
            exempt: Final = scenario.key(
                models=[openai_model, claude_model, deepseek_model], disable_global_guardrails=True
            )
            assert _stored_metadata(exempt)["disable_global_guardrails"] is True
            surfaces: Final = (
                ("chat", openai_model, _chat),
                ("messages", claude_model, _messages),
                ("responses", deepseek_model, _responses),
            )
            for surface, model, call in surfaces:
                for stream in (False, True):
                    marker: Final = f"exempt-{surface}-{stream}-{uuid.uuid4().hex}"
                    response: Final = call(candidate, model, exempt, marker, stream=stream)
                    response.read()
                    assert response.status_code == 200, f"{surface} stream={stream}: {response.text}"
                    assert "synthetic policy denial" not in response.text
                    provider_hits: Final = (
                        _wire_hits(anthropic_wire, marker) if surface == "messages" else _upstream_hits(gateway, marker)
                    )
                    assert provider_hits == 1, f"{surface} stream={stream} marker={marker}"
                    assert _sink_hits(policy, marker) == 0

            base_url: Final = str(candidate.client.base_url).rstrip("/") + "/v1"
            sync_marker: Final = "exempt-sdk-sync-" + uuid.uuid4().hex
            OpenAI(api_key=exempt, base_url=base_url, max_retries=0).chat.completions.create(
                model=openai_model, messages=[{"role": "user", "content": sync_marker}]
            )
            assert _upstream_hits(gateway, sync_marker) == 1

            async_marker: Final = "exempt-sdk-async-" + uuid.uuid4().hex

            async def _async_chat() -> None:
                async with AsyncOpenAI(api_key=exempt, base_url=base_url, max_retries=0) as client:
                    await client.chat.completions.create(
                        model=openai_model, messages=[{"role": "user", "content": async_marker}]
                    )

            asyncio.run(_async_chat())
            assert _upstream_hits(gateway, async_marker) == 1

            anthropic_marker: Final = "exempt-anthropic-" + uuid.uuid4().hex
            Anthropic(
                api_key=exempt, base_url=str(candidate.client.base_url).rstrip("/"), max_retries=0
            ).messages.create(
                model=claude_model, max_tokens=16, messages=[{"role": "user", "content": anthropic_marker}]
            )
            assert _wire_hits(anthropic_wire, anthropic_marker) == 1


@pytest.mark.covers(
    "mgmt.key.disable_global_guardrails.runtime.team_level_flag_exempts_key",
    "mgmt.key.disable_global_guardrails.runtime.resaved_exempt_key_still_exempt",
    "mgmt.key.disable_global_guardrails.runtime.spend_log_exactly_once",
)
def test_team_flag_resaved_key_and_spend_log(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_denying_guardrail) as policy:
        config: Final = _guardrail_config(policy.url, tmp_path / "team-exempt.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")

            exempt_team: Final = scenario.team(models=[model], disable_global_guardrails=True)
            team_key: Final = scenario.key(team_id=exempt_team, models=[model])
            team_marker: Final = "team-exempt-" + uuid.uuid4().hex
            team_response: Final = _chat(candidate, model, team_key, team_marker, stream=False)
            assert team_response.status_code == 200, team_response.text
            assert _upstream_hits(gateway, team_marker) == 1
            assert _sink_hits(policy, team_marker) == 0

            caller_team: Final = scenario.team(
                models=[model], members_with_roles=[{"role": "admin", "user_id": member}]
            )
            admin_exempt: Final = scenario.key(team_id=caller_team, models=[model], disable_global_guardrails=True)
            resave_caller: Final = scenario.key(
                user_id=member, team_id=caller_team, models=[model], allowed_routes=["/key/*", "/v1/chat/completions"]
            )
            resaved: Final = candidate.request(
                "POST",
                "/key/update",
                {
                    "key": admin_exempt,
                    "key_alias": "audit-runtime-resave-" + uuid.uuid4().hex,
                    "metadata": {"disable_global_guardrails": True},
                },
                key=resave_caller,
            )
            assert resaved.status_code == 200, resaved.text
            resave_marker: Final = "resaved-exempt-" + uuid.uuid4().hex
            resave_response: Final = _chat(candidate, model, admin_exempt, resave_marker, stream=False)
            assert resave_response.status_code == 200, resave_response.text
            response_id: Final = string_value(resave_response.json()["id"])
            assert _upstream_hits(gateway, resave_marker) == 1
            assert _sink_hits(policy, resave_marker) == 0
            eventually(
                lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (response_id,)),
                lambda rows: len(rows) == 1,
                seconds=70,
            )
