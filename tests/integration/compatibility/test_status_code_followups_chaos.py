"""The provider goes down in the middle of a concurrent burst across /v1/chat/completions, /v1/messages and
/v1/responses, streaming and not, on a two-worker proxy: every failed call reaches the caller as an error status,
the fix's own 400s keep answering without a provider call, another key on another deployment keeps working, and
once the provider is back every call serves and lands in the spend log exactly once."""

from __future__ import annotations

import json
import os
import re
import signal
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import UpstreamSlot, owned_proxy_process, owned_upstream
from integration._support.upstream import register_scenario
from integration.compatibility._status_code_audit import (
    CHAT,
    CHAT_FRAMES,
    ENDPOINT_BODIES,
    ENDPOINT_PATHS,
    MESSAGE,
    MESSAGE_FRAMES,
    RESPONSE,
    RESPONSES_FRAMES,
    Upstream,
    json_response,
    sse_response,
    stream_finished,
)
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import StoredResponse

pytestmark: Final = pytest.mark.timeout(480)

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_MARKER_HEADER: Final = "x-litellm-spend-logs-metadata"
_KINDS: Final[dict[str, tuple[str, bool, str, str, StoredResponse]]] = {
    "chat": ("chat", False, "chaos-chat", "openai/gpt-4o-mini", json_response(CHAT)),
    "chatstream": ("chat", True, "chaos-chat-stream", "openai/gpt-4o-mini", sse_response(CHAT_FRAMES)),
    "responses": ("responses", False, "chaos-responses", "openai/gpt-4o-mini", json_response(RESPONSE)),
    "responsesstream": (
        "responses",
        True,
        "chaos-responses-stream",
        "openai/gpt-4o-mini",
        sse_response(RESPONSES_FRAMES),
    ),
    "messages": ("messages", False, "chaos-message", "anthropic/claude-haiku-4-5", json_response(MESSAGE)),
    "messagesstream": (
        "messages",
        True,
        "chaos-message-stream",
        "anthropic/claude-haiku-4-5",
        sse_response(MESSAGE_FRAMES),
    ),
}
_BURST: Final = 30


@dataclass(frozen=True, slots=True)
class _Chaos:
    gateway: Gateway
    log: Path
    slot: UpstreamSlot
    identities: Mapping[str, str]
    stable_identity: str
    burst_key: str
    other_key: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    marker: str
    kind: str
    status: int | None
    text: str
    finished: bool = False


def _register_on_slot(slot: UpstreamSlot, identities: Mapping[str, str]) -> None:
    for kind, identity in identities.items():
        register_scenario(identity, _KINDS[kind][4], control_url=slot.url)


@pytest.fixture(scope="module")
def chaos(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Chaos]:
    directory: Final = tmp_path_factory.mktemp("audit-chaos")
    with (
        owned_upstream(directory) as slot,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        identities: Final = {kind: f"chaos-{kind}-{uuid.uuid4().hex}" for kind in _KINDS}
        _register_on_slot(slot, identities)
        stable_identity: Final = f"chaos-stable-{uuid.uuid4().hex}"
        stable_handle: Final = register_scenario(stable_identity, json_response(CHAT))
        model_list: Final[list[JsonValue]] = [
            *(
                {
                    "model_name": _KINDS[kind][2],
                    "litellm_params": {
                        "model": _KINDS[kind][3],
                        "api_base": f"{slot.url}/{identity}",
                        "api_key": identity,
                    },
                }
                for kind, identity in identities.items()
            ),
            {
                "model_name": "*",
                "litellm_params": {"model": "openai/*", "api_base": f"{slot.url}/{identities['chat']}", "api_key": "x"},
            },
            {
                "model_name": "chaos-stable",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": stable_handle.api_base(),
                    "api_key": stable_identity,
                },
            },
        ]
        config: Final = directory / "chaos.yaml"
        config.write_text(
            json.dumps({"model_list": model_list, "router_settings": {"num_retries": 0, "disable_cooldowns": True}}),
            encoding="utf-8",
        )
        burst_key: Final = scenario.key(key_alias=f"chaos-burst-{uuid.uuid4().hex}")
        other_key: Final = scenario.key(key_alias=f"chaos-other-{uuid.uuid4().hex}", models=["chaos-stable"])
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            eventually(lambda: _worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=120)
            yield _Chaos(
                Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url),
                owned.log,
                slot,
                identities,
                stable_identity,
                burst_key,
                other_key,
            )


def _worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(found[1]) for found in _STARTED_WORKER.finditer(log.read_text()))


def _call(chaos: _Chaos, kind: str, marker: str, *, key: str | None = None, without_model: bool = False) -> _Outcome:
    endpoint, stream, model_name, _target, _response = _KINDS[kind]
    body: Final[dict[str, JsonValue]] = {
        **ENDPOINT_BODIES[endpoint],
        **({"stream": True} if stream else {}),
        "model": None if without_model else model_name,
        **({"input": marker} if endpoint == "responses" else {"messages": [{"role": "user", "content": marker}]}),
    }
    headers: Final = {
        "Authorization": f"Bearer {chaos.burst_key if key is None else key}",
        _MARKER_HEADER: json.dumps({"marker": marker}),
    }
    try:
        with httpx.Client(base_url=str(chaos.gateway.client.base_url), trust_env=False, timeout=60) as client:
            with client.stream("POST", ENDPOINT_PATHS[endpoint], json=body, headers=headers) as response:
                lines: Final = tuple(response.iter_lines())
                status: Final = response.status_code
    except httpx.TransportError as error:
        return _Outcome(marker, kind, None, repr(error))
    return _Outcome(marker, kind, status, "\n".join(lines), not stream or stream_finished(endpoint, lines))


def _burst(chaos: _Chaos, phase: str) -> tuple[_Outcome, ...]:
    kinds: Final = tuple(_KINDS)
    plan: Final = tuple(
        (kinds[index % len(kinds)], f"{phase}-{kinds[index % len(kinds)]}-{index}-{uuid.uuid4().hex}")
        for index in range(_BURST)
    )
    with ThreadPoolExecutor(max_workers=12) as pool:
        return tuple(pool.map(lambda item: _call(chaos, item[0], item[1]), plan))


def _missing_model_cells(chaos: _Chaos) -> tuple[_Outcome, ...]:
    return tuple(_call(chaos, kind, f"missing-{kind}-{uuid.uuid4().hex}", without_model=True) for kind in _KINDS)


def _upstream_hits(slot: UpstreamSlot, markers: frozenset[str]) -> Mapping[str, int]:
    upstream: Final = Upstream(slot.url)
    upstream.drain()
    bodies: Final = tuple(json.dumps(item.get("body")) for item in upstream.items)
    return {marker: sum(marker in body for body in bodies) for marker in markers}


def _spend_markers(key: str, markers: frozenset[str]) -> Mapping[str, tuple[str, ...]]:
    rows: Final = read_rows(
        "SELECT status, metadata->'spend_logs_metadata'->>'marker' AS marker "
        'FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
        (sha256(key.encode()).hexdigest(),),
    )
    return {
        marker: found
        for marker in markers
        if (found := tuple(str(row["status"]) for row in rows if str(row["marker"]) == marker))
    }


def _assert_served(outcomes: tuple[_Outcome, ...]) -> None:
    assert [(outcome.kind, outcome.status, outcome.finished) for outcome in outcomes] == [
        (outcome.kind, 200, True) for outcome in outcomes
    ], [(outcome.kind, outcome.status, outcome.text[:300]) for outcome in outcomes if outcome.status != 200]


def _assert_rejected_without_a_model(chaos: _Chaos, outcomes: tuple[_Outcome, ...]) -> None:
    assert [outcome.status for outcome in outcomes] == [400] * len(outcomes), [
        (outcome.kind, outcome.status, outcome.text[:300]) for outcome in outcomes
    ]
    assert all("Invalid model name passed in model=" in outcome.text for outcome in outcomes), outcomes
    assert _upstream_hits(chaos.slot, frozenset(outcome.marker for outcome in outcomes)) == {
        outcome.marker: 0 for outcome in outcomes
    }


def _stable_call(chaos: _Chaos) -> httpx.Response:
    return chaos.gateway.client.post(
        "/v1/chat/completions",
        json={"model": "chaos-stable", "messages": [{"role": "user", "content": "other key"}]},
        headers={"Authorization": f"Bearer {chaos.other_key}"},
        timeout=30,
    )


def _settled(chaos: _Chaos) -> bool:
    return all(_call(chaos, kind, f"settle-{kind}-{uuid.uuid4().hex}").status == 200 for kind in _KINDS)


def _stop_mid_burst(chaos: _Chaos) -> tuple[_Outcome, ...]:
    with ThreadPoolExecutor(max_workers=1) as stopper:
        stopping: Final = stopper.submit(chaos.slot.stop)
        during: Final = _burst(chaos, "during")
        stopping.result(timeout=120)
    return during


def _restart(chaos: _Chaos) -> None:
    chaos.slot.start()
    _register_on_slot(chaos.slot, chaos.identities)
    eventually(lambda: _settled(chaos), lambda ok: ok, seconds=90)


def test_provider_outage_mid_burst_errors_reach_callers_and_recovery_serves_once(chaos: _Chaos) -> None:
    before: Final = _burst(chaos, "before")
    _assert_served(before)

    during: Final = _stop_mid_burst(chaos)
    outage: Final = _burst(chaos, "outage")
    for outcome in outage:
        assert outcome.status is not None and outcome.status >= 500, (outcome.kind, outcome.status, outcome.text)
        assert '"error"' in outcome.text, (outcome.kind, outcome.text)
    for outcome in during:
        assert (outcome.status == 200 and outcome.finished) or (
            outcome.status is not None and outcome.status >= 500 and '"error"' in outcome.text
        ), (outcome.kind, outcome.status, outcome.text[:300])
    assert chaos.gateway.request("GET", "/health/liveliness").status_code == 200
    stable: Final = _stable_call(chaos)
    assert stable.status_code == 200, stable.text

    _restart(chaos)
    after: Final = _burst(chaos, "after")
    _assert_served(after)
    after_markers: Final = frozenset(outcome.marker for outcome in after)
    assert _upstream_hits(chaos.slot, after_markers) == dict.fromkeys(after_markers, 1)

    served: Final = frozenset(outcome.marker for outcome in (*before, *after))
    landed: Final = eventually(
        lambda: _spend_markers(chaos.burst_key, served), lambda found: frozenset(found) == served, seconds=120
    )
    assert landed == dict.fromkeys(served, ("success",)), landed
    assert len(_worker_pids(chaos.log)) >= 2


def test_missing_model_400s_hold_before_during_and_after_an_outage(chaos: _Chaos) -> None:
    _assert_rejected_without_a_model(chaos, _missing_model_cells(chaos))
    _stop_mid_burst(chaos)
    _assert_rejected_without_a_model_while_down(chaos)
    stable: Final = _stable_call(chaos)
    assert stable.status_code == 200, stable.text
    _restart(chaos)
    _assert_rejected_without_a_model(chaos, _missing_model_cells(chaos))


def _assert_rejected_without_a_model_while_down(chaos: _Chaos) -> None:
    outcomes: Final = _missing_model_cells(chaos)
    assert [outcome.status for outcome in outcomes] == [400] * len(outcomes), [
        (outcome.kind, outcome.status, outcome.text[:300]) for outcome in outcomes
    ]
    assert all("Invalid model name passed in model=" in outcome.text for outcome in outcomes), outcomes


def _tolerant_status(chaos: _Chaos, kind: str) -> int | None:
    return _call(chaos, kind, f"kill-probe-{kind}-{uuid.uuid4().hex}").status


def test_killed_worker_leaves_the_sibling_serving_and_the_respawn_rejecting(chaos: _Chaos) -> None:
    workers: Final = _worker_pids(chaos.log)
    assert len(workers) >= 2, workers
    os.kill(workers[-1], signal.SIGKILL)
    eventually(lambda: _tolerant_status(chaos, "chat"), lambda status: status == 200, seconds=60)
    _assert_served(_burst(chaos, "killed"))
    stable: Final = _stable_call(chaos)
    assert stable.status_code == 200, stable.text
    eventually(lambda: _worker_pids(chaos.log), lambda pids: len(pids) == len(workers) + 1, seconds=180)
    eventually(lambda: _settled(chaos), lambda ok: ok, seconds=120)
    _assert_served(_burst(chaos, "respawned"))
    _assert_rejected_without_a_model(chaos, _missing_model_cells(chaos))
