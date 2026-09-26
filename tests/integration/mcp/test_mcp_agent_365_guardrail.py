"""Agent 365 guardrail when its own dependencies fail: Entra and Agent 365 are owned local doubles."""

import json
import os
import signal
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import httpx
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    Outcome,
    echo_tool,
    register_mcp,
    scripted_peer,
    tool_calls,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

TENANT: Final = "00000000-0000-4000-8000-0000000a3650"
EVALUATE_PATH: Final = "/agents/tool-evaluation/evaluate"
CALLER_TOKEN: Final = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnRlZ3JhdGlvbiJ9.synthetic-signature"
GUARDRAIL_TIMEOUT_SECONDS: Final = 1.0
SLOW_REPLY_SECONDS: Final = 3.0
GUARDRAIL_STATUSES: Final = (
    "SELECT metadata->'mcp_tool_call_metadata'->>'name' AS tool, gi->>'guardrail_status' AS status "
    "FROM \"LiteLLM_SpendLogs\", jsonb_array_elements(metadata->'guardrail_information') gi "
    "WHERE api_key = %s AND gi->>'guardrail_provider' = 'agent_365'"
)
GUARDRAIL_ROWS: Final = (
    "SELECT metadata->'guardrail_information' AS gi FROM \"LiteLLM_SpendLogs\" "
    'WHERE api_key = %s AND call_type = %s ORDER BY "startTime"'
)
TOOLS: Final = ("outage", "skipped", "denied", "add", "throttled", "rejected", "nonjson", "nobool", "slow")


def _caller_token(entra_case: str) -> str:
    """A compact JWS whose signature segment tells the Entra double how to answer the OBO exchange."""
    return f"eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnRlZ3JhdGlvbiJ9.{entra_case}"


def _entra(request: Request) -> Reply:
    assert request.target == f"/{TENANT}/oauth2/v2.0/token", request.target
    case: Final = parse_qs(request.body.decode())["assertion"][0].rsplit(".", 1)[-1]
    match case:
        case "entra-outage":
            return Reply(status=503, body=json.dumps({"error": "synthetic Entra outage"}).encode())
        case "entra-nonjson":
            return Reply(body=b"<html>synthetic gateway timeout</html>", content_type="text/html")
        case "entra-slow":
            time.sleep(SLOW_REPLY_SECONDS)
        case "entra-misconfigured":
            misconfigured: Final = {"error": "invalid_client", "error_codes": [7000215]}
            return Reply(status=401, body=json.dumps(misconfigured).encode())
        case "entra-rejected":
            rejected: Final = {"error": "invalid_grant", "error_codes": [50013]}
            return Reply(status=400, body=json.dumps(rejected).encode())
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
        case "throttled":
            return Reply(status=429, body=json.dumps({"error": "synthetic throttle"}).encode())
        case "rejected":
            return Reply(status=400, body=json.dumps({"error": "synthetic malformed evaluation request"}).encode())
        case "nonjson":
            return Reply(body=b"<html>synthetic upstream error page</html>", content_type="text/html")
        case "nobool":
            return Reply(body=json.dumps({"allowed": "yes", "defender": {"status": "Evaluated"}}).encode())
        case "slow":
            time.sleep(SLOW_REPLY_SECONDS)
    return Reply(body=json.dumps({"allowed": True, "defender": {"status": "Evaluated"}}).encode())


def _generic_guardrail_outage(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api", request.target
    return Reply(status=503, body=json.dumps({"error": "synthetic sibling guardrail outage"}).encode())


def _config(
    tmp_path: Path,
    name: str,
    entra_url: str | None,
    agent_365_url: str,
    fallback: str | None,
    sibling_url: str | None,
) -> Path:
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
                "api_base": agent_365_url,
                "timeout": GUARDRAIL_TIMEOUT_SECONDS,
                **({"authority_host": entra_url} if entra_url else {}),
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
    caller: McpCaller
    key: str
    alias: str
    guardrail_name: str
    peer: McpPeer
    entra: Wire
    agent_365: Wire
    server_id: str

    def caller_for(self, entry: EntryPoint, token: str = CALLER_TOKEN) -> McpCaller:
        return McpCaller(self.candidate, self.key, entry, self.alias, headers={"Authorization": f"Bearer {token}"})

    def catalog_by_worker(self, samples: int = 8) -> frozenset[tuple[int, bool]]:
        """A fresh connection stays on the worker that accepted it, so its self-reported pid pairs with its catalog."""

        def probe(connection: httpx.Client) -> tuple[int, bool] | None:
            candidate: Final = replace(self.candidate, client=connection)
            caller: Final = McpCaller(
                candidate, self.key, "mcp", self.alias, headers={"Authorization": f"Bearer {CALLER_TOKEN}"}
            )
            try:
                summary: Final = connection.get(
                    "/debug/memory/summary", headers={"Authorization": f"Bearer {candidate.key}"}
                )
                return int(summary.json()["worker_pid"]), f"{self.alias}-add" in caller.list_tools().tools
            except httpx.TransportError:
                return None

        with ExitStack() as connections, ThreadPoolExecutor(max_workers=samples) as pool:
            fresh: Final = tuple(
                connections.enter_context(httpx.Client(base_url=str(self.candidate.client.base_url)))
                for _ in range(samples)
            )
            return frozenset(seen for seen in pool.map(probe, fresh) if seen is not None)

    def every_worker_serves_the_catalog(self, workers: int, without: int | None = None) -> frozenset[int]:
        seen: Final = eventually(
            self.catalog_by_worker,
            lambda pairs: (
                len({pid for pid, _ in pairs}) == workers
                and without not in {pid for pid, _ in pairs}
                and all(served for _, served in pairs)
            ),
            seconds=40,
        )
        return frozenset(pid for pid, _ in seen)

    def call_without_bearer(self, tool: str) -> Outcome:
        return McpCaller(self.candidate, self.key, "mcp", self.alias).call(f"{self.alias}-{tool}", {"a": 0})

    def upstream_tool_names(self) -> tuple[str, ...]:
        return tuple(str(call["body"]["params"]["name"]) for call in tool_calls(self.peer.drain()))


@contextmanager
def _rig(
    gateway: Gateway,
    tmp_path: Path,
    fallback: str | None,
    *,
    workers: int = 1,
    environment: dict[str, str] | None = None,
    authority_env: str | None = None,
    sibling: bool = False,
) -> Iterator[Rig]:
    """``authority_env`` names the environment variable that carries the Entra double instead of the config."""
    alias: Final = "a365" + uuid.uuid4().hex[:8]
    with (
        wire_server(_entra) as entra,
        wire_server(_agent_365) as agent_365,
        wire_server(_generic_guardrail_outage) as sibling_outage,
        scripted_peer(*(echo_tool(tool) for tool in TOOLS)) as peer,
        owned_proxy_process(
            gateway,
            tmp_path,
            {
                "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2",
                **(environment or {}),
                **({authority_env: entra.url} if authority_env else {}),
            },
            config=_config(
                tmp_path,
                alias,
                None if authority_env else entra.url,
                agent_365.url,
                fallback,
                sibling_outage.url if sibling else None,
            ),
            workers=workers,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(owned.gateway, key, "mcp", alias, headers={"Authorization": f"Bearer {CALLER_TOKEN}"})
        rig: Final = Rig(owned.gateway, caller, key, alias, alias, peer, entra, agent_365, identity)
        if workers > 1:
            rig.every_worker_serves_the_catalog(workers)
        peer.drain()
        yield rig


def _guardrail_statuses(key: str) -> dict[str, str]:
    rows: Final = read_rows(GUARDRAIL_STATUSES, (sha256(key.encode()).hexdigest(),))
    return {str(row["tool"]): str(row["status"]) for row in rows}


def _chat(rig: Rig, model: str, marker: str) -> httpx.Response:
    return rig.candidate.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}, key=rig.key
    )


def test_default_lets_the_call_through_unscanned_when_agent_365_cannot_evaluate(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        outage: Final = rig.caller.call(f"{rig.alias}-outage", {"a": 1})
        assert outage.text == '{"a": 1}', f"Agent 365 down must fail open by default: {outage.raw}"
        skipped: Final = rig.caller.call(f"{rig.alias}-skipped", {"a": 2})
        assert skipped.text == '{"a": 2}', f"Defender skipping the call must fail open by default: {skipped.raw}"
        denied: Final = rig.caller.call(f"{rig.alias}-denied", {"a": 3})
        assert denied.error is not None and "Blocked by Microsoft Defender" in denied.raw, denied.raw
        assert rig.upstream_tool_names() == ("outage", "skipped")
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 3, seconds=70)
        assert statuses == {
            "outage": "guardrail_failed_to_respond",
            "skipped": "guardrail_failed_to_respond",
            "denied": "guardrail_intervened",
        }, statuses


def test_explicit_fail_closed_blocks_with_503_and_never_reaches_upstream_when_agent_365_is_down(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback="fail_closed") as rig:
        outage: Final = rig.caller.call(f"{rig.alias}-outage", {"a": 1})
        assert outage.error is not None and "could not authorize the tool call" in outage.raw, outage.raw
        assert rig.caller.call(f"{rig.alias}-add", {"a": 2}).text == '{"a": 2}'
        assert rig.upstream_tool_names() == ("add",)
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 2, seconds=70)
        assert statuses == {"outage": "guardrail_failed_to_respond", "add": "success"}, statuses


def test_explicit_fail_open_matches_the_default_and_still_blocks_policy_denials(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback="fail_open") as rig:
        assert rig.caller.call(f"{rig.alias}-outage", {"a": 1}).text == '{"a": 1}'
        denied: Final = rig.caller.call(f"{rig.alias}-denied", {"a": 2})
        assert denied.error is not None and "Blocked by Microsoft Defender" in denied.raw, denied.raw
        assert rig.upstream_tool_names() == ("outage",)
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 2, seconds=70)
        assert statuses == {"outage": "guardrail_failed_to_respond", "denied": "guardrail_intervened"}, statuses


def test_default_fails_open_on_malformed_or_stalled_agent_365_replies(gateway: Gateway, tmp_path: Path) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        for index, tool in enumerate(("nonjson", "nobool", "slow")):
            outcome: Final = rig.caller.call(f"{rig.alias}-{tool}", {"a": index})
            assert outcome.text == json.dumps({"a": index}), f"{tool}: {outcome.raw}"
        assert rig.upstream_tool_names() == ("nonjson", "nobool", "slow")
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 3, seconds=70)
        assert statuses == dict.fromkeys(("nonjson", "nobool", "slow"), "guardrail_failed_to_respond"), statuses


def test_default_fails_open_when_entra_is_down_stalled_malformed_or_refuses_the_gateway_credentials(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        cases: Final = ("entra-outage", "entra-slow", "entra-nonjson", "entra-misconfigured")
        for index, case in enumerate(cases):
            outcome: Final = rig.caller_for("mcp", _caller_token(case)).call(f"{rig.alias}-add", {"a": index})
            assert outcome.text == json.dumps({"a": index}), f"{case}: {outcome.raw}"
        assert rig.upstream_tool_names() == ("add",) * len(cases)
        assert tuple(
            sorted(parse_qs(request.body.decode())["assertion"][0].rsplit(".", 1)[-1] for request in rig.entra.drain())
        ) == tuple(sorted(cases))
        assert rig.agent_365.drain() == (), "no OBO token means no evaluation request"
        rows: Final = eventually(
            lambda: read_rows(GUARDRAIL_ROWS, (sha256(rig.key.encode()).hexdigest(), "call_mcp_tool")),
            lambda seen: len(seen) >= len(cases),
            seconds=70,
        )
        assert [row["gi"][0]["guardrail_status"] for row in rows] == ["guardrail_failed_to_respond"] * len(cases)


def test_throttling_and_ordinary_4xx_from_agent_365_keep_blocking_under_the_default(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        throttled: Final = rig.caller.call(f"{rig.alias}-throttled", {"a": 1})
        assert throttled.error == "Error: Agent 365 guardrail could not authorize the tool call", throttled.raw
        rejected: Final = rig.caller.call(f"{rig.alias}-rejected", {"a": 2})
        assert rejected.error == "Error: Agent 365 rejected the tool evaluation request", rejected.raw
        assert rig.upstream_tool_names() == ()
        statuses: Final = eventually(lambda: _guardrail_statuses(rig.key), lambda seen: len(seen) >= 2, seconds=70)
        assert statuses == {"throttled": "guardrail_failed_to_respond", "rejected": "guardrail_intervened"}, statuses


def test_caller_authentication_failures_keep_blocking_under_the_default(gateway: Gateway, tmp_path: Path) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        rejected: Final = "Error: Agent 365 guardrail rejected the tool call"
        missing: Final = rig.call_without_bearer("add")
        assert missing.error == rejected, missing.raw
        malformed: Final = rig.caller_for("mcp", "not-a-jws").call(f"{rig.alias}-add", {"a": 1})
        assert malformed.error == rejected, malformed.raw
        refused: Final = rig.caller_for("mcp", _caller_token("entra-rejected")).call(f"{rig.alias}-add", {"a": 2})
        assert refused.error == rejected, refused.raw
        assert len(rig.entra.drain()) == 1, "only the well-formed bearer reaches the OBO exchange"
        assert rig.upstream_tool_names() == ()
        assert rig.agent_365.drain() == ()
        rows: Final = eventually(
            lambda: read_rows(GUARDRAIL_ROWS, (sha256(rig.key.encode()).hexdigest(), "call_mcp_tool")),
            lambda seen: len(seen) >= 3,
            seconds=70,
        )
        assert [row["gi"][0]["guardrail_status"] for row in rows] == ["guardrail_intervened"] * 3


def test_every_mcp_entry_point_fails_open_on_outage_and_blocks_denials(gateway: Gateway, tmp_path: Path) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig:
        for entry in ENTRY_POINTS:
            caller: Final = rig.caller_for(entry)
            passed: Final = caller.call(f"{rig.alias}-outage", {"entry": entry}, server_id=rig.server_id)
            assert passed.text == json.dumps({"entry": entry}), f"{entry}: {passed.raw}"
            denied: Final = caller.call(f"{rig.alias}-denied", {"entry": entry}, server_id=rig.server_id)
            assert denied.error is not None and "Blocked by Microsoft Defender" in denied.raw, f"{entry}: {denied.raw}"
        assert rig.upstream_tool_names() == ("outage",) * len(ENTRY_POINTS)


def test_chat_completions_never_touch_agent_365_while_a_sibling_guardrail_keeps_its_fail_closed_default(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None, sibling=True) as rig, rig.candidate.scenario() as scenario:
        model: Final = scenario.model()
        chat: Final = _chat(rig, model, "sibling-" + uuid.uuid4().hex)
        assert chat.status_code == 500 and "Generic Guardrail API failed" in chat.text, chat.text
        assert rig.entra.drain() == () and rig.agent_365.drain() == ()
        assert rig.caller.call(f"{rig.alias}-outage", {"a": 1}).text == '{"a": 1}'
        assert rig.upstream_tool_names() == ("outage",)


def test_chat_completions_are_unaffected_by_the_mcp_guardrail(gateway: Gateway, tmp_path: Path) -> None:
    with _rig(gateway, tmp_path, fallback=None) as rig, rig.candidate.scenario() as scenario:
        model: Final = scenario.model()
        marker: Final = "unaffected-" + uuid.uuid4().hex
        chat: Final = _chat(rig, model, marker)
        assert chat.status_code == 200, chat.text
        assert rig.entra.drain() == () and rig.agent_365.drain() == ()
        rows: Final = eventually(
            lambda: read_rows(GUARDRAIL_ROWS, (sha256(rig.key.encode()).hexdigest(), "acompletion")),
            lambda seen: len(seen) >= 1,
            seconds=70,
        )
        assert rows[0]["gi"] is None, rows


def test_agent365_authority_host_env_wins_over_azure_authority_host_when_config_has_none(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path / "azure", fallback=None, authority_env="AZURE_AUTHORITY_HOST") as azure_only:
        assert azure_only.caller.call(f"{azure_only.alias}-add", {"a": 1}).text == '{"a": 1}'
        assert len(azure_only.entra.drain()) == 1
    with (
        wire_server(_entra) as decoy,
        _rig(
            gateway,
            tmp_path / "agent365",
            fallback=None,
            authority_env="AGENT365_AUTHORITY_HOST",
            environment={"AZURE_AUTHORITY_HOST": decoy.url},
        ) as both,
    ):
        assert both.caller.call(f"{both.alias}-add", {"a": 2}).text == '{"a": 2}'
        assert len(both.entra.drain()) == 1 and decoy.drain() == ()


def test_thirty_call_burst_against_a_flapping_agent_365_reaches_upstream_exactly_once_each_on_two_workers(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None, workers=2) as rig:
        markers: Final = tuple(("outage" if index % 2 else "add", uuid.uuid4().hex) for index in range(30))
        with ThreadPoolExecutor(max_workers=10) as pool:
            outcomes: Final = tuple(
                pool.map(lambda pair: rig.caller.call(f"{rig.alias}-{pair[0]}", {"marker": pair[1]}), markers)
            )
        for (tool, marker), outcome in zip(markers, outcomes, strict=True):
            assert outcome.text == json.dumps({"marker": marker}), f"{tool}: {outcome.raw}"
        seen: Final = sorted(
            str(call["body"]["params"]["arguments"]["marker"]) for call in tool_calls(rig.peer.drain())
        )
        assert seen == sorted(marker for _, marker in markers)
        assert rig.caller.call(f"{rig.alias}-denied", {"a": 1}).error is not None
        assert rig.upstream_tool_names() == ()


def test_default_survives_a_worker_kill_and_keeps_blocking_denials_on_two_workers(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, fallback=None, workers=2) as rig:
        before: Final = rig.every_worker_serves_the_catalog(2)
        victim: Final = min(before)
        os.kill(victim, signal.SIGKILL)
        after: Final = rig.every_worker_serves_the_catalog(2, without=victim)
        assert after - before, f"a replacement worker took over: before {before}, after {after}"
        rig.peer.drain()
        for index in range(10):
            passed: Final = rig.caller.call(f"{rig.alias}-outage", {"a": index})
            assert passed.text == json.dumps({"a": index}), passed.raw
            denied: Final = rig.caller.call(f"{rig.alias}-denied", {"a": index})
            assert denied.error is not None and "Blocked by Microsoft Defender" in denied.raw, denied.raw
        assert rig.upstream_tool_names() == ("outage",) * 10
