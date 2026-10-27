"""Slots C1, C2, C3 and D5: callback credentials must reach only their sink.

C1 is the team callback ``langfuse_secret_key`` (team callback API, the deprecated team
``metadata.callback_settings`` and the config ``default_team_settings``), C2 the key-level
``metadata.logging`` Langfuse key, C3 a team callback ``dd_api_key`` for Datadog, and D5 a
``langfuse_secret_key`` the caller sends in the request body (``langfuse_host`` in a body is
rejected without an admin opt-in, so D5 runs on its own proxy with
``general_settings.allow_client_side_credentials`` on).

Positive control: the owning sink double must receive the request's marker under an auth
header built from the canary (Langfuse ``Basic pk:sk``, Datadog ``DD-API-KEY``), or the test
fails before sweeping. Sensitivity control: the marker must be seen in the stored request body,
the Logs drawer route and the owning sink. Then no sweep may find the canary anywhere else,
including every request the provider double received (swept as the ``provider`` sink, with no
header allowance; the provider's own key is slot B1, which these tests do not search for).
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import quote

import pytest
from integration._support.client import Scenario
from integration._support.wire import Request, wire_server
from integration.security._callback_traffic import (
    ENDPOINTS,
    EXPECTED_STATUS,
    LANGFUSE_PUBLIC_KEY,
    OUTCOMES,
    datadog_sink,
    langfuse_sink,
    outcome_text,
    send,
    spend_request_id,
    upstream,
    wait_for_sink,
)
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Caller, Recorder, Rig, canary_rig
from integration.security._sweeps import assert_marker_seen, assert_no_hits, record_route_sweep, sweep_all
from pydantic import JsonValue

LANGFUSE: Final = "langfuse"
DATADOG: Final = "datadog"
PROVIDER: Final = "provider"
BOTH: Final = "success_and_failure"


@dataclass(frozen=True, slots=True)
class CallbackRig:
    rig: Rig
    langfuse: Recorder
    datadog: Recorder

    def sinks(self) -> dict[str, tuple[Request, ...]]:
        return {
            **{name: sink.requests() for name, sink in self.rig.sinks.items()},
            LANGFUSE: self.langfuse.requests(),
            DATADOG: self.datadog.requests(),
            PROVIDER: self.rig.provider.requests(),
        }

    def datadog_port(self) -> str:
        return self.datadog.url.rsplit(":", 1)[1]


@contextmanager
def callback_rig(
    root: Path, configure: Callable[[dict[str, object], str, str], None] | None = None
) -> Iterator[CallbackRig]:
    with (
        wire_server(langfuse_sink) as langfuse,
        wire_server(datadog_sink) as datadog,
        canary_rig(
            root,
            configure=(lambda config, provider: configure(config, provider, langfuse.url)) if configure else None,
            environment={"LANGFUSE_FLUSH_INTERVAL": "1"},
            upstream=upstream,
        ) as rig,
    ):
        yield CallbackRig(rig, Recorder(langfuse), Recorder(datadog))


def _allow_client_side_credentials(config: dict[str, object], _provider: str, _langfuse: str) -> None:
    settings: Final = config["general_settings"]
    assert isinstance(settings, dict)
    settings["allow_client_side_credentials"] = True


@pytest.fixture(scope="module")
def client_side(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(tmp_path_factory.mktemp("canary-client-side"), _allow_client_side_credentials) as value:
        yield value


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(tmp_path_factory.mktemp("canary-callbacks")) as value:
        yield value


def langfuse_vars(secret: Canary, host: str) -> dict[str, JsonValue]:
    return {"langfuse_public_key": LANGFUSE_PUBLIC_KEY, "langfuse_secret_key": secret.value, "langfuse_host": host}


def caller(
    scenario: Scenario,
    *,
    team_id: str | None = None,
    team_metadata: Mapping[str, JsonValue] | None = None,
    key_metadata: Mapping[str, JsonValue] | None = None,
) -> Caller:
    team: Final = scenario.team(
        **({"team_id": team_id} if team_id else {}), **({"metadata": dict(team_metadata)} if team_metadata else {})
    )
    user: Final = scenario.member(team)
    key: Final = scenario.key(
        team_id=team, user_id=user, models=[CONFIG_MODEL], **({"metadata": dict(key_metadata)} if key_metadata else {})
    )
    return Caller(team, user, key)


def langfuse_control(secret: Canary) -> Callable[[CallbackRig, Canary], None]:
    expected: Final = "Basic " + base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{secret.value}".encode()).decode()

    def check(rig: CallbackRig, marker: Canary) -> None:
        delivered: Final = wait_for_sink(rig.langfuse, marker)
        assert {request.headers.get("authorization") for request in delivered} == {expected}, (
            f"Positive control: the Langfuse double never received the {secret.slot} canary as its Basic auth"
        )

    return check


def datadog_control(secret: Canary) -> Callable[[CallbackRig, Canary], None]:
    def check(rig: CallbackRig, marker: Canary) -> None:
        delivered: Final = wait_for_sink(rig.datadog, marker)
        assert {request.headers.get("dd-api-key") for request in delivered} == {secret.value}, (
            "Positive control: the Datadog double never received the C3 canary as DD-API-KEY"
        )

    return check


def run_scenario(
    cb: CallbackRig,
    scenario: Scenario,
    who: Caller,
    secret: Canary,
    endpoint: str,
    outcome: str,
    *,
    control: Callable[[CallbackRig, Canary], None],
    sink: str,
    own_header: tuple[str, str],
    node: str,
    extra: Mapping[str, JsonValue] | None = None,
) -> None:
    marker: Final = canary(MARKER)
    started: Final = datetime.now(UTC)
    response: Final = send(
        cb.rig.proxy, who.key, endpoint, CONFIG_MODEL, outcome_text(secret.slot, marker, outcome), extra
    )
    assert response.status_code == EXPECTED_STATUS[outcome], response.text
    control(cb, marker)
    request_id: Final = spend_request_id(marker)
    wait_for_sink(cb.rig.sinks[GENERIC_SINK], marker)

    report: Final = sweep_all(
        cb.rig.proxy,
        (marker, secret),
        responses=(response,),
        sinks=cb.sinks(),
        ids={
            "request_id": request_id,
            "team_id": who.team_id,
            "user_id": who.user_id,
            "model_id": cb.rig.model_id,
            "model": CONFIG_MODEL,
        },
        callers=who.callers(cb.rig),
        own_headers={**cb.rig.own_headers, sink: own_header},
        since=started,
    )
    record_route_sweep(report.routes, node)
    assert_marker_seen(
        report,
        {
            "S1": "LiteLLM_SpendLogs.proxy_server_request",
            "S2": f"GET /spend/logs/ui/{quote(request_id, safe='')} as admin -> 200",
            "S4": f"{sink}[",
        },
    )
    assert_marker_seen(report, {"S2": f"GET /spend/logs?request_id={quote(request_id, safe='')} as admin -> 200"})
    assert_marker_seen(report, {"S4": f"{PROVIDER}["})
    assert_no_hits(report.credential_hits(), f"slot {secret.slot}, {endpoint}, {outcome}")


MATRIX: Final = [
    pytest.param(endpoint, outcome, id=f"{endpoint}-{outcome}") for endpoint in ENDPOINTS for outcome in OUTCOMES
]


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~430 GET routes as two callers
@pytest.mark.parametrize(("endpoint", "outcome"), MATRIX)
def test_c1_team_callback_api_langfuse_secret_reaches_only_langfuse(
    shared: CallbackRig, endpoint: str, outcome: str, request: pytest.FixtureRequest
) -> None:
    secret: Final = canary("C1")
    with shared.rig.proxy.scenario() as scenario:
        who: Final = caller(scenario)
        shared.rig.proxy.post(
            f"/team/{who.team_id}/callback",
            {
                "callback_name": "langfuse",
                "callback_type": BOTH,
                "callback_vars": langfuse_vars(secret, shared.langfuse.url),
            },
        )
        run_scenario(
            shared,
            scenario,
            who,
            secret,
            endpoint,
            outcome,
            control=langfuse_control(secret),
            sink=LANGFUSE,
            own_header=("authorization", "C1"),
            node=request.node.nodeid,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~430 GET routes as two callers
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_c1_deprecated_team_callback_settings_langfuse_secret_reaches_only_langfuse(
    shared: CallbackRig, endpoint: str, request: pytest.FixtureRequest
) -> None:
    secret: Final = canary("C1")
    settings: Final = {
        "success_callback": ["langfuse"],
        "failure_callback": ["langfuse"],
        "callback_vars": langfuse_vars(secret, shared.langfuse.url),
    }
    with shared.rig.proxy.scenario() as scenario:
        who: Final = caller(scenario, team_metadata={"callback_settings": settings})
        run_scenario(
            shared,
            scenario,
            who,
            secret,
            endpoint,
            "success",
            control=langfuse_control(secret),
            sink=LANGFUSE,
            own_header=("authorization", "C1"),
            node=request.node.nodeid,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~430 GET routes as two callers
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_c1_config_default_team_settings_langfuse_secret_reaches_only_langfuse(
    tmp_path: Path, endpoint: str, request: pytest.FixtureRequest
) -> None:
    """The team callback comes from ``litellm_settings.default_team_settings`` in config.yaml."""
    secret: Final = canary("C1")
    team_id: Final = f"canary-config-team-{secret.core[:12]}"

    def configure(config: dict[str, object], _provider: str, langfuse_url: str) -> None:
        settings: Final = config["litellm_settings"]
        assert isinstance(settings, dict)
        settings["default_team_settings"] = [
            {
                "team_id": team_id,
                "success_callback": ["langfuse"],
                "failure_callback": ["langfuse"],
                "langfuse_public_key": LANGFUSE_PUBLIC_KEY,
                "langfuse_secret": secret.value,
                "langfuse_host": langfuse_url,
            }
        ]

    with callback_rig(tmp_path, configure) as cb, cb.rig.proxy.scenario() as scenario:
        who: Final = caller(scenario, team_id=team_id)
        run_scenario(
            cb,
            scenario,
            who,
            secret,
            endpoint,
            "success",
            control=langfuse_control(secret),
            sink=LANGFUSE,
            own_header=("authorization", "C1"),
            node=request.node.nodeid,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~430 GET routes as two callers
@pytest.mark.parametrize(("endpoint", "outcome"), MATRIX)
def test_c2_key_logging_langfuse_secret_reaches_only_langfuse(
    shared: CallbackRig, endpoint: str, outcome: str, request: pytest.FixtureRequest
) -> None:
    secret: Final = canary("C2")
    logging: Final = [
        {
            "callback_name": "langfuse",
            "callback_type": BOTH,
            "callback_vars": langfuse_vars(secret, shared.langfuse.url),
        }
    ]
    with shared.rig.proxy.scenario() as scenario:
        who: Final = caller(scenario, key_metadata={"logging": logging})
        run_scenario(
            shared,
            scenario,
            who,
            secret,
            endpoint,
            outcome,
            control=langfuse_control(secret),
            sink=LANGFUSE,
            own_header=("authorization", "C2"),
            node=request.node.nodeid,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~430 GET routes as two callers
@pytest.mark.parametrize(("endpoint", "outcome"), MATRIX)
def test_c3_team_callback_datadog_api_key_reaches_only_datadog(
    shared: CallbackRig, endpoint: str, outcome: str, request: pytest.FixtureRequest
) -> None:
    secret: Final = canary("C3")
    with shared.rig.proxy.scenario() as scenario:
        who: Final = caller(scenario)
        shared.rig.proxy.post(
            f"/team/{who.team_id}/callback",
            {
                "callback_name": "datadog",
                "callback_type": BOTH,
                "callback_vars": {
                    "dd_api_key": secret.value,
                    "dd_agent_host": "127.0.0.1",
                    "dd_agent_port": shared.datadog_port(),
                },
            },
        )
        run_scenario(
            shared,
            scenario,
            who,
            secret,
            endpoint,
            outcome,
            control=datadog_control(secret),
            sink=DATADOG,
            own_header=("dd-api-key", "C3"),
            node=request.node.nodeid,
        )


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~430 GET routes as two callers
@pytest.mark.parametrize(("endpoint", "outcome"), MATRIX)
def test_d5_request_body_langfuse_secret_reaches_only_langfuse(
    client_side: CallbackRig, endpoint: str, outcome: str, request: pytest.FixtureRequest
) -> None:
    secret: Final = canary("D5")
    with client_side.rig.proxy.scenario() as scenario:
        who: Final = caller(scenario)
        run_scenario(
            client_side,
            scenario,
            who,
            secret,
            endpoint,
            outcome,
            control=langfuse_control(secret),
            sink=LANGFUSE,
            own_header=("authorization", "D5"),
            node=request.node.nodeid,
            extra={
                **langfuse_vars(secret, client_side.langfuse.url),
                "success_callback": ["langfuse"],
                "failure_callback": ["langfuse"],
            },
        )


def test_find_canary_sees_the_langfuse_basic_auth_header() -> None:
    """The Langfuse positive control and own-header rule depend on decoding ``Basic pk:sk``."""
    secret: Final = canary("C1")
    header: Final = "Basic " + base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{secret.value}".encode()).decode()
    assert [match.slot for match in find_canary(header, (secret,))] == ["C1"]
