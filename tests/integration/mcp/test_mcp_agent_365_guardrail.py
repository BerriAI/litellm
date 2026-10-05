"""Agent 365 guardrail paths that end before the OBO exchange, so neither Entra nor Agent 365 is ever contacted."""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    echo_tool,
    register_mcp,
    scripted_peer,
    tool_calls,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

TENANT: Final = "00000000-0000-4000-8000-0000000a3650"
REJECTED: Final = "Agent 365 guardrail rejected the tool call"
GUARDRAIL_ROWS: Final = (
    "SELECT COALESCE(jsonb_path_query_first(metadata, "
    "'$.guardrail_information[*] ? (@.guardrail_name == $name).guardrail_status', "
    "jsonb_build_object('name', %s::text)) #>> '{}', 'none') AS status "
    'FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s ORDER BY "startTime"'
)
FALLBACKS: Final = (None, "fail_open", "fail_closed")


def _generic_guardrail_outage(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api", request.target
    return Reply(status=503, body=json.dumps({"error": "synthetic sibling guardrail outage"}).encode())


def _config(tmp_path: Path, name: str, fallback: str | None, sibling_url: str | None) -> Path:
    config: dict = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": name,
            "litellm_params": {
                "guardrail": "agent_365",
                "mode": "pre_mcp_call",
                "default_on": True,
                "tenant_id": TENANT,
                "client_id": "synthetic-client-id",
                "client_secret": "synthetic-client-secret",
                **({"unreachable_fallback": fallback} if fallback else {}),
            },
        },
        *(
            [
                {
                    "guardrail_name": f"{name}-sibling",
                    "litellm_params": {
                        "guardrail": "generic_guardrail_api",
                        "mode": "pre_call",
                        "default_on": True,
                        "api_base": sibling_url,
                    },
                }
            ]
            if sibling_url
            else []
        ),
    ]
    path: Final = tmp_path / "agent_365.yaml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    candidate: Gateway
    key: str
    alias: str
    peer: McpPeer
    server_id: str

    def caller(self, entry: EntryPoint = "mcp", bearer: str | None = None) -> McpCaller:
        headers: Final = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        return McpCaller(self.candidate, self.key, entry, self.alias, headers=headers)

    def upstream_tool_names(self) -> tuple[str, ...]:
        return tuple(str(call["body"]["params"]["name"]) for call in tool_calls(self.peer.drain()))

    def guardrail_statuses(self, call_type: str, at_least: int) -> list[str]:
        rows: Final = eventually(
            lambda: read_rows(GUARDRAIL_ROWS, (self.alias, sha256(self.key.encode()).hexdigest(), call_type)),
            lambda seen: len(seen) >= at_least,
            seconds=70,
        )
        return [str(row["status"]) for row in rows]


@contextmanager
def _rig(gateway: Gateway, tmp_path: Path, fallback: str | None, *, sibling: bool = False) -> Iterator[Rig]:
    alias: Final = "a365" + uuid.uuid4().hex[:8]
    with (
        wire_server(_generic_guardrail_outage) as sibling_outage,
        scripted_peer(echo_tool("add")) as peer,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"},
            config=_config(tmp_path, alias, fallback, sibling_outage.url if sibling else None),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        peer.drain()
        yield Rig(owned.gateway, key, alias, peer, identity)


def _chat(rig: Rig, model: str, marker: str) -> httpx.Response:
    return rig.candidate.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}, key=rig.key
    )


@pytest.mark.parametrize("fallback", FALLBACKS)
def test_a_missing_or_malformed_caller_bearer_blocks_on_every_entry_point_whatever_the_fallback(
    gateway: Gateway, tmp_path: Path, fallback: str | None
) -> None:
    with _rig(gateway, tmp_path, fallback) as rig:
        assert f"{rig.alias}-add" in rig.caller().list_tools().tools, "the catalog needs only the virtual key"
        for entry in ENTRY_POINTS:
            missing: Final = rig.caller(entry).call(f"{rig.alias}-add", {"entry": entry}, server_id=rig.server_id)
            assert missing.error is not None and REJECTED in missing.raw, f"{entry} without a bearer: {missing.raw}"
            malformed: Final = rig.caller(entry, "not-a-jws").call(
                f"{rig.alias}-add", {"entry": entry}, server_id=rig.server_id
            )
            assert malformed.error is not None and REJECTED in malformed.raw, f"{entry} opaque bearer: {malformed.raw}"
        assert rig.upstream_tool_names() == ()
        expected: Final = 2 * len(ENTRY_POINTS)
        assert rig.guardrail_statuses("call_mcp_tool", expected) == ["guardrail_intervened"] * expected


def test_chat_completions_are_unaffected_by_the_mcp_guardrail(gateway: Gateway, tmp_path: Path) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig, rig.candidate.scenario() as scenario:
        model: Final = scenario.model()
        chat: Final = _chat(rig, model, "unaffected-" + uuid.uuid4().hex)
        assert chat.status_code == 200, chat.text
        assert rig.guardrail_statuses("acompletion", 1) == ["none"]


def test_a_sibling_guardrail_keeps_its_own_fail_closed_default_next_to_agent_365(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None, sibling=True) as rig, rig.candidate.scenario() as scenario:
        model: Final = scenario.model()
        chat: Final = _chat(rig, model, "sibling-" + uuid.uuid4().hex)
        assert chat.status_code == 500 and "Generic Guardrail API failed" in chat.text, chat.text
