"""Slot G1 through a Datadog intake double: the sink key reaches only its own auth header.

The owned proxy enables the ``datadog`` callback with ``DD_API_KEY`` set to a fresh G1 canary
and ``DD_BASE_URL`` pointed at a local intake double. Datadog batches are gzip-compressed JSON
(a single event sent on the sync path is plain JSON), so the double inflates ``Content-Encoding:
gzip`` bodies, requires JSON log events, answers 202 like the real intake, and records the bytes
exactly as received for S4 (``find_canary`` inflates them). Events the route sweep itself
produces are swept again after it.

Positive control: the intake double must receive ``DD-API-KEY: <G1 canary>`` on the batch
carrying the scenario's marker, and the provider double ``Authorization: Bearer <B1 canary>``.
Sensitivity control: the marker must be found inside the gzip body (encoding ``gzip``), in the
stored spend row, on the Logs drawer route and in the generic sink. Then S1 to S5 plus the
intake double may not hold B1 or G1 anywhere, except G1 in the intake's own ``dd-api-key`` and
on the proxy admin's callback settings route (``ADMIN_ONLY_ALLOWANCES``). That route's gate for
everyone else is asserted directly: the internal user gets 401, and a ``proxy_admin_viewer``
must read ``DD_API_KEY`` as ``REDACTED``. Routes are swept as the admin, the internal user and
that admin viewer.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually, string_value
from integration._support.wire import Reply, Request, wire_server
from integration.security._canary import MARKER, Canary, canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Recorder, Rig, canary_rig, settle, team_caller
from integration.security._sweeps import (
    assert_marker_seen,
    assert_no_hits,
    record_route_sweep,
    sweep_all,
    sweep_sink,
)

DATADOG_SINK: Final = "datadog"
DATADOG_KEY_HEADER: Final = "dd-api-key"
CALLBACK_SETTINGS_ROUTE: Final = "/get/config/callbacks"


def inflated(request: Request) -> bytes:
    """The body as Datadog reads it: batches are gzip-compressed, single sync events are not."""
    return gzip.decompress(request.body) if request.headers.get("content-encoding") == "gzip" else request.body


def datadog_intake(request: Request) -> Reply:
    assert request.target == "/api/v2/logs", request.target
    events: Final = json.loads(inflated(request))
    assert isinstance(events, (list, dict)) and events, events
    return Reply(status=202, body=b"{}")


def enable_datadog(config: dict[str, object], _provider_url: str) -> None:
    settings: Final = config["litellm_settings"]
    assert isinstance(settings, dict)
    settings["callbacks"] = [*settings["callbacks"], DATADOG_SINK]


@pytest.fixture
def intake() -> Iterator[Recorder]:
    with wire_server(datadog_intake) as wire:
        yield Recorder(wire)


@pytest.fixture
def g1() -> Canary:
    return canary("G1")


@pytest.fixture
def rig(tmp_path: Path, intake: Recorder, g1: Canary) -> Iterator[Rig]:
    environment: Final = {"DD_API_KEY": g1.value, "DD_SITE": "datadog.invalid", "DD_BASE_URL": intake.url}
    with canary_rig(tmp_path, configure=enable_datadog, environment=environment) as value:
        yield value


def carrying_inflated(intake: Recorder, marker: Canary) -> tuple[Request, ...]:
    """Gzip batches whose inflated body holds ``marker``."""
    return tuple(
        request
        for request in intake.requests()
        if request.headers.get("content-encoding") == "gzip" and marker.core.encode() in inflated(request)
    )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as three callers
def test_datadog_api_key_reaches_only_its_own_header(
    rig: Rig, intake: Recorder, g1: Canary, request: pytest.FixtureRequest
) -> None:
    b1: Final = rig.canaries["B1"]
    marker: Final = canary(MARKER)
    started: Final = datetime.now(UTC)
    with rig.proxy.scenario() as scenario:
        caller: Final = team_caller(scenario)
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"slot G1 {marker.value}"}]},
            key=caller.key,
        )
        assert response.status_code == 200, response.text
        assert [request.headers.get("authorization") for request in rig.provider.carrying(marker.value)] == [
            f"Bearer {b1.value}"
        ], "Positive control: the provider double never received the B1 canary"
        request_id: Final = string_value(response.json()["id"])
        settle(rig, request_id, marker)
        batches: Final = eventually(lambda: carrying_inflated(intake, marker), bool, seconds=30)
        assert {batch.headers.get(DATADOG_KEY_HEADER) for batch in batches} == {g1.value}, (
            "Positive control: the Datadog intake double never received the G1 canary"
        )
        assert all(marker.core.encode() not in batch.body for batch in batches), "Datadog body was not compressed"

        denied: Final = rig.proxy.request("GET", CALLBACK_SETTINGS_ROUTE, key=caller.key)
        assert denied.status_code == 401, f"internal_user read the callback settings: {denied.text}"
        viewer: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin_viewer"))
        settings: Final = rig.proxy.request("GET", CALLBACK_SETTINGS_ROUTE, key=viewer)
        assert settings.status_code == 200, settings.text
        datadog_variables: Final = [
            entry["variables"] for entry in settings.json()["callbacks"] if entry["name"] == DATADOG_SINK
        ]
        assert datadog_variables and all(variables["DD_API_KEY"] == "REDACTED" for variables in datadog_variables), (
            f"The admin viewer's callback settings did not redact DD_API_KEY: {datadog_variables}"
        )

        swept: Final = intake.requests()
        report: Final = sweep_all(
            rig.proxy,
            (marker, b1, g1),
            responses=(response,),
            sinks={**{name: sink.requests() for name, sink in rig.sinks.items()}, DATADOG_SINK: swept},
            ids={
                "request_id": request_id,
                "team_id": caller.team_id,
                "user_id": caller.user_id,
                "model_id": CONFIG_MODEL,
                "model": CONFIG_MODEL,
            },
            callers={**caller.callers(rig), "admin_viewer": viewer},
            own_headers={**rig.own_headers, DATADOG_SINK: (DATADOG_KEY_HEADER, "G1")},
            since=started,
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
        assert_marker_seen(report, {"S2": f"GET /spend/logs?request_id={request_id} as admin -> 200"})
        assert any(
            hit.slot == MARKER and hit.location.startswith(f"{DATADOG_SINK}[") and hit.encoding == "gzip"
            for hit in report.hits
        ), f"Sensitivity control: S4 never inflated the marker out of the Datadog body: {report.marker_locations()}"
        late: Final = sweep_sink(
            f"{DATADOG_SINK} after the route sweep",
            intake.requests()[len(swept) :],
            (b1, g1),
            own_header=(DATADOG_KEY_HEADER, "G1"),
        )
        assert_no_hits((*report.credential_hits(), *late), "slots B1 and G1, Datadog intake")
