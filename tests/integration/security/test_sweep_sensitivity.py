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
from integration.security._canary import MARKER, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Rig, canary_rig, settle, team_caller
from integration.security._sweeps import assert_marker_seen, record_route_sweep, sweep_all, sweep_redis


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
    assert find_canary(f"Basic {basic}".replace(basic[10:20], "A" * 10), (marker,)) == ()
    assert find_canary(f"sk-...{marker.core[-4:]}", (marker,)) == ()


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
