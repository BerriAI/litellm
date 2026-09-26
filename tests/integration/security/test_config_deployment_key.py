"""Slot B1: a deployment ``api_key`` declared in the proxy config reaches only the provider.

Positive control: the provider double must receive ``Authorization: Bearer <B1 canary>`` for
the scenario's request, or the test fails before sweeping. Sensitivity control: the marker sent
in the same request must be reported by the sweeps where stored prompts belong. Then no sweep
may find the B1 canary anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import string_value
from integration.security._canary import MARKER, canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, PROVIDER_4XX, Rig, canary_rig, settle, team_caller
from integration.security._sweeps import assert_marker_seen, assert_no_hits, record_route_sweep, sweep_all


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    """One owned proxy per test: B1 lives in the config, so a fresh core needs a fresh proxy."""
    with canary_rig(tmp_path) as value:
        yield value


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as two callers
@pytest.mark.parametrize("outcome", ["success", "provider_4xx"])
def test_config_deployment_api_key_reaches_only_the_provider(
    rig: Rig, outcome: str, request: pytest.FixtureRequest
) -> None:
    b1: Final = rig.canaries["B1"]
    marker: Final = canary(MARKER)
    text: Final = f"slot B1 {marker.value}" + (f" {PROVIDER_4XX}" if outcome == "provider_4xx" else "")
    with rig.proxy.scenario() as scenario:
        caller: Final = team_caller(scenario)
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": text}]},
            key=caller.key,
        )
        assert response.status_code == (200 if outcome == "success" else 400), response.text
        delivered: Final = rig.provider.carrying(marker.value)
        assert [request.headers.get("authorization") for request in delivered] == [f"Bearer {b1.value}"], (
            "Positive control: the provider double never received the B1 canary"
        )
        request_id: Final = (
            string_value(response.json()["id"]) if outcome == "success" else response.headers["x-litellm-call-id"]
        )
        settle(rig, request_id, marker)

        report: Final = sweep_all(
            rig.proxy,
            (marker, b1),
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
                "S4": f"{GENERIC_SINK}[",
            },
        )
        assert_no_hits(report.credential_hits(), f"slot B1, {outcome}")
