"""Sensitivity controls: every sweep must find a marker where prompts are legitimately stored.

A sweep that cannot see its surface would pass every credential slot vacuously. Each test here
sends a fresh marker in message content with ``store_prompts_in_spend_logs`` on and requires
each sweep to report it at the place it belongs.
"""

from __future__ import annotations

import base64
import gzip
import json
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
import pytest
from redis import Redis
from integration._support.client import Gateway, eventually, string_value
from integration.security._canary import DECODE_BUDGET_BYTES, MARKER, SLOTS, DecodeBudgetExceeded, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Rig, canary_rig, chat_upstream, settle, team_caller
from integration._support.wire import Reply, Request, wire_server
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.tls import server_context, write_self_signed_cert
from tests.integration._support.workers import worker_services
from integration.security._sweeps import (
    ADMIN_ONLY_ALLOWANCES,
    ALLOWANCE_SLOT_FAMILIES,
    PROVIDER_PASSTHROUGH_REASON,
    assert_marker_seen,
    get_routes,
    record_route_sweep,
    route_allowance,
    route_denied,
    scoped_queries,
    sweep_all,
    sweep_redis,
    sweep_routes,
    sweep_sink,
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with canary_rig(tmp_path_factory.mktemp("canary-sensitivity")) as value:
        yield value


@pytest.mark.parametrize("prefix", ["", "u:", "us:", "use:"], ids=["align0", "align1", "align2", "align3"])
def test_find_canary_decodes_base64_at_every_alignment_and_gzip(prefix: str) -> None:
    marker: Final = canary(MARKER)
    basic: Final = base64.b64encode(f"{prefix}{marker.value}".encode()).decode()
    urlsafe: Final = base64.urlsafe_b64encode(f"{prefix}{marker.value}".encode()).decode().rstrip("=")
    assert [match.slot for match in find_canary(f"Authorization: Basic {basic}", (marker,))] == [MARKER]
    assert [match.slot for match in find_canary(f'{{"token":"{urlsafe}"}}', (marker,))] == [MARKER]
    assert [match.slot for match in find_canary(gzip.compress(f"Basic {basic}".encode()), (marker,))] == [MARKER]
    embedded: Final = b"prefix:" + gzip.compress(f"Basic {basic}".encode()) + b":suffix"
    assert [match.slot for match in find_canary(embedded, (marker,))] == [MARKER]
    members: Final = gzip.compress(b"first member") + gzip.compress(f"Basic {basic}".encode())
    assert [match.slot for match in find_canary(members, (marker,))] == [MARKER]
    binary_wrapper: Final = bytes(range(256)) + f" Basic {basic} ".encode() + bytes(range(256))
    assert [match.slot for match in find_canary(base64.b64encode(binary_wrapper), (marker,))] == [MARKER]
    assert find_canary(f"Basic {basic}".replace(basic[10:20], "A" * 10), (marker,)) == ()
    assert find_canary(f"sk-...{marker.core[-4:]}", (marker,)) == ()


def test_find_canary_fails_loudly_past_its_decode_budget() -> None:
    marker: Final = canary(MARKER)
    bomb: Final = gzip.compress(b"\0" * (1024 * 1024 + 1))
    with pytest.raises(DecodeBudgetExceeded):
        find_canary(bomb, (marker,), budget_bytes=1024 * 1024)
    assert find_canary(gzip.compress(b"\0" * 1024) + marker.value.encode(), (marker,), budget_bytes=1024 * 1024)
    assert DECODE_BUDGET_BYTES >= 256 * 1024 * 1024


def test_rig_with_an_overridden_master_key_resolves_the_config_deployment(tmp_path: Path) -> None:
    master_key: Final = f"sk-canary-override-{uuid.uuid4().hex}"
    with canary_rig(tmp_path, environment={"LITELLM_MASTER_KEY": master_key}) as overridden:
        assert overridden.proxy.key == master_key
        assert overridden.model_id
        assert overridden.proxy.request("GET", "/model/info").status_code == 200


def _skills_upstream(request: Request) -> Reply:
    skill: Final = {
        "id": "owned-skill",
        "display_title": "Owned skill",
        "source": "custom",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    if request.target.startswith("/v1/skills/"):
        return Reply(body=json.dumps(skill).encode())
    if request.target.startswith("/v1/skills"):
        return Reply(body=json.dumps({"data": [skill], "has_more": False}).encode())
    return chat_upstream(request)


def test_skills_reads_use_the_owned_provider_despite_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://ambient-provider.invalid")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ambient-provider.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-provider-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-provider-token")
    with canary_rig(tmp_path, upstream=_skills_upstream) as owned:
        for path in ("/v1/skills", "/v1/skills/owned-skill"):
            response: Final = owned.proxy.request("GET", path)
            assert response.status_code == 200, response.text
        requests: Final = owned.provider.requests()
        assert tuple(request.target.split("?", 1)[0] for request in requests) == (
            "/v1/skills",
            "/v1/skills/owned-skill",
        )
        assert all(request.headers.get("x-api-key") == owned.canaries["B1"].value for request in requests)
        assert all("authorization" not in request.headers for request in requests)
        assert owned.egress() == ()


def test_route_allowances_match_only_their_exact_route_and_caller() -> None:
    routes: Final = get_routes()
    callers: Final = ("admin", "internal_user", "Admin", "admin ", "")
    for route, caller in ADMIN_ONLY_ALLOWANCES:
        assert route in routes, f"Allowance names a route the proxy no longer registers: {route}"
        for variant in (route + "/", route.upper(), route.rstrip("s"), "/v1" + route):
            assert route_allowance(variant, caller) is None, variant
    allowed: Final = {(route, caller) for route in routes for caller in callers if route_allowance(route, caller)}
    assert allowed == set(ADMIN_ONLY_ALLOWANCES), allowed
    assert all(route_denied(route) is None for route, _ in ADMIN_ONLY_ALLOWANCES)
    assert set(ALLOWANCE_SLOT_FAMILIES) == set(ADMIN_ONLY_ALLOWANCES)
    for (route, caller), families in ALLOWANCE_SLOT_FAMILIES.items():
        for family in families:
            assert route_allowance(route, caller, family + "1") is not None
        for slot in SLOTS:
            if not slot.startswith(families):
                assert route_allowance(route, caller, slot) is None, (route, caller, slot)


def test_only_provider_passthrough_routes_match_the_passthrough_deny_rule() -> None:
    denied: Final = {route for route in get_routes() if route_denied(route) == PROVIDER_PASSTHROUGH_REASON}
    assert "/openai/{endpoint:path}" in denied and "/langfuse/{endpoint:path}" in denied
    assert all(route.endswith("/{endpoint:path}") and route.count("{") == 1 for route in denied), denied
    for swept in ("/v1/files/{file_id:path}", "/spend/logs/ui/{request_id}", "/v1/memory/{key:path}"):
        assert route_denied(swept) is None, swept


def test_sink_own_header_allows_only_that_header_and_slot() -> None:
    own: Final = canary("B1")
    other: Final = canary(MARKER)
    request: Final = Request(
        "POST",
        "/",
        {"authorization": f"Bearer {own.value}", "x-extra": f"Bearer {own.value}", "x-other": other.value},
        f'{{"copied": "{own.value}"}}'.encode(),
    )
    hits: Final = sweep_sink("double", (request,), (own, other), own_header=("authorization", own.slot))
    assert {(hit.slot, hit.location) for hit in hits} == {
        (MARKER, "double[0] POST / header x-other"),
        ("B1", "double[0] POST / body"),
        ("B1", "double[0] POST / header x-extra"),
    }


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as two callers
def test_every_sweep_finds_the_stored_prompt_marker(rig: Rig, request: pytest.FixtureRequest) -> None:
    marker: Final = canary(MARKER)
    started: Final = datetime.now(UTC)
    with rig.proxy.scenario() as scenario:
        caller: Final = team_caller(scenario)
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"sensitivity {marker.value}"}]},
            key=caller.key,
        )
        assert response.status_code == 200, response.text
        assert len(rig.provider.carrying(marker.value)) == 1
        request_id: Final = string_value(response.json()["id"])
        settle(rig, request_id, marker)
        eventually(lambda: sweep_redis((marker,)), bool, seconds=10)

        ids: Final = {
            "request_id": request_id,
            "team_id": caller.team_id,
            "user_id": caller.user_id,
            "model_id": rig.model_id,
            "model": CONFIG_MODEL,
        }
        report: Final = sweep_all(
            rig.proxy,
            (marker,),
            responses=(response,),
            sinks={name: sink.requests() for name, sink in rig.sinks.items()},
            ids=ids,
            callers=caller.callers(rig),
            own_headers=rig.own_headers,
            since=started,
        )
        record_route_sweep(report.routes, request.node.nodeid)
        assert_marker_seen(
            report,
            {
                "S1": "LiteLLM_SpendLogs.proxy_server_request",
                "S2": f"GET /spend/logs/ui/{request_id} as admin -> 200",
                "S3": "response[0] POST /v1/chat/completions -> 200 body",
                "S4": f"{GENERIC_SINK}[",
                "S5": "redis value",
            },
        )
        assert_marker_seen(report, {"S2": f"GET /spend/logs?request_id={request_id} as admin -> 200"})
        assert_marker_seen(report, {"S2": f"GET /spend/logs?user_id={caller.user_id} as admin -> 200"})
        assert_marker_seen(report, {"S2": f"GET /spend/logs/ui/{request_id} as internal_user -> 200"})
        for route in ("/spend/logs/ui", "/spend/logs/v2"):
            filtered = tuple(query for query in scoped_queries(route, ids, started) if "_id=" in query)
            assert len(filtered) == 2, filtered
            for query in filtered:
                assert report.routes.statuses.get(f"GET {route}{query} as admin") == 200, (route, query)
                listed = rig.proxy.request("GET", route + query)
                assert request_id in listed.text, f"{route}{query} does not list the scenario's row"
        assert report.credential_hits() == ()


def test_security_worker_services_isolate_identical_database_and_cache_keys(tmp_path: Path) -> None:
    with (
        worker_services(tmp_path / "first") as first,
        worker_services(tmp_path / "second") as second,
    ):
        for environment, marker in ((first, "first"), (second, "second")):
            write_rows(
                "CREATE TABLE IF NOT EXISTS isolation_probe (id text PRIMARY KEY, marker text NOT NULL)",
                (),
                database_url=environment["DATABASE_URL"],
            )
            write_rows(
                "INSERT INTO isolation_probe VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET marker=EXCLUDED.marker",
                ("same-id", marker),
                database_url=environment["DATABASE_URL"],
            )
            with Redis(host=environment["REDIS_HOST"], port=int(environment["REDIS_PORT"])) as cache:
                cache.set("same-key", marker)
        for environment, marker in ((first, "first"), (second, "second")):
            assert read_rows(
                "SELECT marker FROM isolation_probe WHERE id=%s",
                ("same-id",),
                database_url=environment["DATABASE_URL"],
            ) == [{"marker": marker}]
            with Redis(
                host=environment["REDIS_HOST"], port=int(environment["REDIS_PORT"]), decode_responses=True
            ) as cache:
                assert cache.get("same-key") == marker


def test_route_sweep_keeps_connections_auth_and_response_cookies_isolated() -> None:
    callers: Final = {"admin": "sk-sweep-admin", "internal_user": "sk-sweep-user"}
    with wire_server(_response_with_cookie, keep_alive=True) as peer:
        with httpx.Client(base_url=peer.url, trust_env=False) as client:
            report: Final = sweep_routes(Gateway(client, callers["admin"], peer.url), (), {}, callers=callers)
        requests: Final = peer.drain()
        requested: Final = tuple(called.split(" ", 1) for called in report.called)
        expected: Final = Counter((path, f"Bearer {callers[caller]}") for caller, path in requested)
        assert report.called
        assert len(requests) == len(report.called)
        assert Counter((request.target, request.headers["authorization"]) for request in requests) == expected
        assert all("cookie" not in request.headers for request in requests)
        assert peer.connections() == len(requests)
        assert report.errors == report.unreachable == ()


def _response_with_cookie(_: Request) -> Reply:
    return Reply(headers={"set-cookie": "session=another-caller; Path=/"})


def test_route_sweep_rejects_an_untrusted_https_peer(tmp_path: Path) -> None:
    cert, key = write_self_signed_cert(tmp_path)
    with wire_server(_response_with_cookie, tls=server_context(cert, key), keep_alive=True) as peer:
        with httpx.Client(base_url=peer.url, trust_env=False) as client:
            report: Final = sweep_routes(Gateway(client, "sk-sweep-admin", peer.url), (), {})
        assert report.called
        assert len(report.unreachable) == len(report.called)
        assert peer.drain() == ()


def test_route_sweep_preserves_truncated_response_errors_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("INTEGRATION_RESULTS_DIR", str(tmp_path))
    with (
        owned_redis(tmp_path) as cache,
        wire_server(lambda _: Reply(chunks=(b"first", b"missing"), abort_after=1)) as peer,
    ):
        monkeypatch.setenv("REDIS_HOST", cache.host)
        monkeypatch.setenv("REDIS_PORT", str(cache.port))
        with (
            httpx.Client(base_url=peer.url, trust_env=False) as client,
            pytest.raises(AssertionError, match="GET routes returned no response"),
        ):
            sweep_all(Gateway(client, "sk-sweep-admin", peer.url), (), responses=(), sinks={}, ids={})
        received: Final = peer.drain()

    saved: Final = json.loads((tmp_path / "security-route-sweep-failures.jsonl").read_text())
    assert received
    assert request.node.nodeid in saved["node"]
    assert saved["called"] == len(received)
    assert len(saved["unreachable"]) == len(received)
    assert all(error.partition("RemoteProtocolError: ")[2] for error in saved["unreachable"])
    assert not (tmp_path / "security-route-sweep.jsonl").exists()
