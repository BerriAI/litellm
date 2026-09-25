"""Agent 365 guardrail when its own dependencies fail: Entra and Agent 365 are owned local doubles."""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpCaller, McpPeer, echo_tool, register_mcp, scripted_peer, tool_calls
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from prometheus_client.parser import text_string_to_metric_families

TENANT: Final = "00000000-0000-4000-8000-0000000a3650"
EVALUATE_PATH: Final = "/agents/tool-evaluation/evaluate"
GUARDRAIL_ERRORS: Final = "litellm_guardrail_errors_total"
CALLER_TOKEN: Final = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnRlZ3JhdGlvbiJ9.synthetic-signature"
GUARDRAIL_STATUSES: Final = (
    "SELECT metadata->'mcp_tool_call_metadata'->>'name' AS tool, gi->>'guardrail_status' AS status "
    "FROM \"LiteLLM_SpendLogs\", jsonb_array_elements(metadata->'guardrail_information') gi "
    "WHERE api_key = %s AND gi->>'guardrail_provider' = 'agent_365'"
)


def _entra(request: Request) -> Reply:
    assert request.target == f"/{TENANT}/oauth2/v2.0/token", request.target
    return Reply(body=json.dumps({"access_token": "obo-" + uuid.uuid4().hex, "expires_in": 3599}).encode())


def _agent_365(request: Request) -> Reply:
    assert request.target == EVALUATE_PATH, request.target
    tool: Final = str(json.loads(request.body)["tool"]["name"]).rsplit("-", 1)[-1]
    match tool:
        case "outage":
            return Reply(status=503, body=json.dumps({"error": "synthetic Agent 365 outage"}).encode())
        case "skipped":
            return Reply(body=json.dumps({"allowed": True, "defender": {"status": "Skipped"}}).encode())
        case "denied":
            verdict: Final = {"status": "Evaluated", "verdict": "Block", "message": "synthetic block"}
            return Reply(body=json.dumps({"allowed": False, "defender": verdict, "correlationId": "denied-1"}).encode())
        case _:
            return Reply(body=json.dumps({"allowed": True, "defender": {"status": "Evaluated"}}).encode())


def _config(tmp_path: Path, name: str, entra_url: str, agent_365_url: str, fallback: str | None) -> Path:
    config: dict = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["callbacks"] = ["prometheus"]
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
                "api_base": agent_365_url,
                "authority_host": entra_url,
                **({"unreachable_fallback": fallback} if fallback else {}),
            },
        }
    ]
    path: Final = tmp_path / "agent_365.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    candidate: Gateway
    caller: McpCaller
    key: str
    alias: str
    guardrail_name: str
    peer: McpPeer


@contextmanager
def _rig(gateway: Gateway, tmp_path: Path, fallback: str | None) -> Iterator[Rig]:
    alias: Final = "a365" + uuid.uuid4().hex[:8]
    prom_dir: Final = tmp_path / "prom"
    prom_dir.mkdir()
    with (
        wire_server(_entra) as entra,
        wire_server(_agent_365) as agent_365,
        scripted_peer(echo_tool("outage"), echo_tool("skipped"), echo_tool("denied"), echo_tool("add")) as peer,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir)},
            config=_config(tmp_path, alias, entra.url, agent_365.url, fallback),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(owned.gateway, key, "mcp", alias, headers={"Authorization": f"Bearer {CALLER_TOKEN}"})
        peer.drain()
        yield Rig(owned.gateway, caller, key, alias, alias, peer)


def _guardrail_statuses(key: str) -> dict[str, str]:
    rows: Final = read_rows(GUARDRAIL_STATUSES, (sha256(key.encode()).hexdigest(),))
    return {str(row["tool"]): str(row["status"]) for row in rows}


def _guardrail_error_counts(candidate: Gateway, guardrail_name: str) -> dict[str, float]:
    response: Final = candidate.client.get(
        "/metrics", headers={"Authorization": f"Bearer {candidate.key}"}, follow_redirects=True
    )
    assert response.status_code == 200, f"GET /metrics: {response.status_code} {response.text[:300]}"
    return {
        sample.labels["error_type"]: float(sample.value)
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == GUARDRAIL_ERRORS and sample.labels.get("guardrail_name") == guardrail_name
    }


def test_default_lets_the_call_through_unscanned_and_counts_it_when_agent_365_cannot_evaluate(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        outage: Final = rig.caller.call(f"{rig.alias}-outage", {"a": 1})
        assert outage.text == '{"a": 1}', f"Agent 365 down must fail open by default: {outage.raw}"
        skipped: Final = rig.caller.call(f"{rig.alias}-skipped", {"a": 2})
        assert skipped.text == '{"a": 2}', f"Defender skipping the call must fail open by default: {skipped.raw}"
        denied: Final = rig.caller.call(f"{rig.alias}-denied", {"a": 3})
        assert denied.error is not None and "Blocked by Microsoft Defender" in denied.raw, denied.raw
        assert tuple(call["body"]["params"]["name"] for call in tool_calls(rig.peer.drain())) == ("outage", "skipped")
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 3, seconds=70)
        assert statuses == {
            "outage": "guardrail_failed_to_respond",
            "skipped": "guardrail_failed_to_respond",
            "denied": "guardrail_intervened",
        }, statuses
        counts: Final = eventually(
            lambda: _guardrail_error_counts(rig.candidate, rig.guardrail_name), lambda seen: len(seen) >= 2
        )
        assert counts == {"fail_open": 2.0, "HTTPException": 1.0}, counts


def test_explicit_fail_closed_blocks_with_503_and_never_reaches_upstream_when_agent_365_is_down(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback="fail_closed") as rig:
        outage: Final = rig.caller.call(f"{rig.alias}-outage", {"a": 1})
        assert outage.error is not None and "could not authorize the tool call" in outage.raw, outage.raw
        assert rig.caller.call(f"{rig.alias}-add", {"a": 2}).text == '{"a": 2}'
        assert tuple(call["body"]["params"]["name"] for call in tool_calls(rig.peer.drain())) == ("add",)
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 2, seconds=70)
        assert statuses == {"outage": "guardrail_failed_to_respond", "add": "success"}, statuses
        assert _guardrail_error_counts(rig.candidate, rig.guardrail_name) == {"HTTPException": 1.0}
