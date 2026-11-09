"""One hook per request under a concurrent burst while the logger's destination stalls and fails, and while a
worker is killed."""

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.hook_counts import (
    STREAMING_SURFACES,
    SURFACE_CALL_TYPES,
    HookRecords,
    assert_hooks,
    one_async_hook,
    probe_hooks,
    provider,
    send,
    write_hook_count_config,
)
from integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 300)

RecordProperty = Callable[[str, object], None]

_SETTINGS: Final = {"callbacks": ["integration._support.hook_count_probe.probe"]}
_HOLD_SECONDS: Final = 2.0


@dataclass(frozen=True)
class _Call:
    surface: str
    outcome: str
    stream: bool
    call_id: str


@dataclass(frozen=True)
class _Outcome:
    call: _Call
    status: int | None
    error: str | None


def _burst(rounds: int) -> tuple[_Call, ...]:
    shapes: Final = (
        *((surface, outcome, False) for surface in sorted(SURFACE_CALL_TYPES) for outcome in ("success", "failure")),
        *((surface, "success", True) for surface in STREAMING_SURFACES),
    )
    return tuple(
        _Call(surface, outcome, stream, str(uuid.uuid4())) for _ in range(rounds) for surface, outcome, stream in shapes
    )


def _attempt(proxy: Gateway, call: _Call) -> _Outcome:
    try:
        response: Final = send(proxy, call.surface, call.outcome, call.call_id, stream=call.stream)
    except httpx.HTTPError as error:
        return _Outcome(call, None, repr(error))
    return _Outcome(call, response.status_code, None)


def _expected_status(call: _Call) -> int:
    return 200 if call.outcome == "success" else 400


def _done(futures: tuple[Future[_Outcome], ...]) -> int:
    return sum(future.done() for future in futures)


class _Destination:
    """The probe's sink: answers at once, holds each report until released, or refuses it with a 503."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.release = threading.Event()

    def __call__(self, request: Request) -> Reply:
        if self.mode == "hold":
            self.release.wait(timeout=_HOLD_SECONDS + 1)
        if self.mode == "fail":
            return Reply(status=503, body=b'{"error":"destination unavailable"}')
        return Reply()


def test_logger_destination_stalling_then_failing_mid_burst_gets_one_hook_per_request(
    tmp_path: Path, record_property: RecordProperty
) -> None:
    destination: Final = _Destination()
    with (
        wire_server(provider) as upstream,
        wire_server(destination) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(tmp_path / "hook-count-chaos.yaml", upstream.url, _SETTINGS)
        with owned_proxy(
            rig, tmp_path, {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as proxy:
            records: Final = HookRecords(sink, function_callbacks=False)
            calls: Final = _burst(2)
            destination.mode = "hold"
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                futures: Final = tuple(pool.submit(_attempt, proxy, call) for call in calls)
                eventually(lambda: _done(futures), lambda done: done >= 1, seconds=60)
                held_liveliness: Final = proxy.client.get("/health/liveliness").status_code
                time.sleep(_HOLD_SECONDS)
                destination.mode = "fail"
                destination.release.set()
                time.sleep(_HOLD_SECONDS)
                destination.mode = "ok"
                outcomes: Final = tuple(future.result(timeout=180) for future in futures)
            record_property("burst_size", len(calls))
            assert held_liveliness == 200, held_liveliness
            assert [outcome for outcome in outcomes if outcome.status != _expected_status(outcome.call)] == []
            for call in calls:
                assert_hooks(records, call.surface, call.call_id, one_async_hook(records, call.surface, call.outcome))


def test_worker_killed_mid_burst_leaves_at_most_one_hook_per_request(
    tmp_path: Path, record_property: RecordProperty
) -> None:
    with (
        wire_server(provider) as upstream,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = write_hook_count_config(tmp_path / "hook-count-kill.yaml", upstream.url, _SETTINGS)
        with owned_proxy_process(
            rig, tmp_path, {"HOOK_COUNT_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as owned:
            proxy: Final = owned.gateway
            records: Final = HookRecords(sink, function_callbacks=False)
            victim: Final = min(eventually(owned.worker_pids, lambda found: len(found) == 2, seconds=60))
            calls: Final = _burst(2)
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                futures: Final = tuple(pool.submit(_attempt, proxy, call) for call in calls)
                eventually(lambda: _done(futures), lambda done: done >= 1, seconds=60)
                with owned.replacing_worker(victim) as kill:
                    kill()
                    done_at_kill: Final = _done(futures)
                    survived: Final = eventually(
                        lambda: _attempt(proxy, _Call("speech", "success", False, str(uuid.uuid4()))),
                        lambda outcome: outcome.status == 200,
                        seconds=30,
                    )
                outcomes: Final = tuple(future.result(timeout=180) for future in futures)
            respawned: Final = _Call("speech", "success", False, str(uuid.uuid4()))
            after_respawn: Final = _attempt(proxy, respawned)
            completed: Final = tuple(outcome for outcome in outcomes if outcome.status is not None)
            interrupted: Final = tuple(outcome for outcome in outcomes if outcome.status is None)
            assert after_respawn.status == 200, after_respawn
            assert [outcome for outcome in completed if outcome.status != _expected_status(outcome.call)] == []
            for call in (survived.call, respawned):
                assert_hooks(records, call.surface, call.call_id, one_async_hook(records, call.surface, call.outcome))
            burst_hooks: Final = held(
                lambda: _hooks_of(records, calls),
                lambda found: all(
                    hooks in ([], [f"async_{call.outcome}"]) for call, hooks in zip(calls, found, strict=True)
                ),
                holding=3,
                seconds=20,
            )
            record_property("burst_size", len(calls))
            record_property("killed_worker_pid", victim)
            record_property("done_at_kill", done_at_kill)
            record_property("interrupted_calls", str(tuple(outcome.error for outcome in interrupted)))
            record_property("burst_calls_without_a_hook", sum(not hooks for hooks in burst_hooks))


def _hooks_of(records: HookRecords, calls: tuple[_Call, ...]) -> tuple[list[str], ...]:
    """A burst call the killed worker served may lose its logging with the worker, whether or not its response
    arrived first, so a burst call gets its one async hook or none, never two."""
    return tuple(probe_hooks(records, call.surface, records.for_call(call.call_id)) for call in calls)
