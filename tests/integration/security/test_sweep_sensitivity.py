"""Sensitivity controls: every sweep must find a marker where prompts are legitimately stored.

A sweep that cannot see its surface would pass every credential slot vacuously. Each test here
sends a fresh marker in message content with ``store_prompts_in_spend_logs`` on and requires
each sweep to report it at the place it belongs.
"""

from __future__ import annotations

import base64
import gzip
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually, string_value
from integration.security._canary import DECODE_BUDGET_BYTES, MARKER, DecodeBudgetExceeded, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Rig, canary_rig, settle, team_caller
from integration._support.wire import Request
from integration.security._sweeps import (
    ADMIN_ONLY_ALLOWANCES,
    PROVIDER_PASSTHROUGH_REASON,
    assert_marker_seen,
    get_routes,
    record_route_sweep,
    route_allowance,
    route_denied,
    sweep_all,
    sweep_redis,
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

        report: Final = sweep_all(
            rig.proxy,
            (marker,),
            responses=(response,),
            sinks={name: sink.requests() for name, sink in rig.sinks.items()},
            ids={"request_id": request_id, "team_id": caller.team_id, "model_id": CONFIG_MODEL, "model": CONFIG_MODEL},
            callers=caller.callers(rig),
            own_headers=rig.own_headers,
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
        assert_marker_seen(report, {"S2": "GET /spend/logs as admin -> 200"})
        assert_marker_seen(report, {"S2": f"GET /spend/logs/ui/{request_id} as internal_user -> 200"})
        assert report.credential_hits() == ()
