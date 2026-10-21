from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.small_bodies_5rps import CASE
from memory_readings import RssCapture
from setup.observe import Observation, evaluate, junit_properties
from setup.performance_test_case import Limit, Ramp, ramped
from setup.plan import build_plan
from setup.ramp import StepReading, send_offsets, step_properties, sustained_rps
from setup.wire import WireRecord

NO_RSS: Final = RssCapture(readings=(), failures=())


def test_send_offsets_put_each_step_s_requests_inside_its_window_in_plan_order() -> None:
    ramp: Final = Ramp(first_rps=10.0, last_rps=30.0, step_rps=10.0, step_seconds=10.0, held_p99_seconds=5.0)
    offsets: Final = send_offsets(ramp, seed=13)
    assert ramp.counts == (100, 200, 300)
    assert len(offsets) == ramp.total_requests
    assert offsets == tuple(sorted(offsets))
    for seq, offset in enumerate(offsets, start=1):
        step = ramp.step_of(seq)
        assert step * 10.0 <= offset < (step + 1) * 10.0, f"request {seq} at {offset:.2f}s is outside step {step}"
    assert send_offsets(ramp, seed=13) == offsets
    assert send_offsets(ramp, seed=14) != offsets


def test_build_plan_carries_the_schedule_only_for_a_ramped_case() -> None:
    ramp: Final = Ramp(first_rps=2.0, last_rps=4.0, step_rps=2.0, step_seconds=5.0, held_p99_seconds=5.0)
    ramped_case: Final = replace(CASE, traffic=ramped(CASE.traffic, ramp, seed=3, concurrency=20))
    assert ramped_case.traffic.expected_seconds == 10.0
    assert [item.send_at_s for item in build_plan(ramped_case)] == list(send_offsets(ramp, seed=3))
    assert all(item.send_at_s is None for item in build_plan(CASE))


def _step(rps: float, p99: float, completed_rps: float | None = None, unexpected: int = 0) -> StepReading:
    return StepReading(
        rps=rps,
        planned=int(rps * 90),
        completed_rps=rps if completed_rps is None else completed_rps,
        gateway_added_p99=p99,
        unexpected=unexpected,
        held_p99_seconds=5.0,
    )


def test_sustained_rps_is_the_last_step_held_before_the_first_that_broke() -> None:
    assert sustained_rps((_step(10, 0.5), _step(20, 1.0), _step(30, 9.0), _step(40, 0.7))) == 20
    assert sustained_rps((_step(10, 0.5), _step(20, 1.0, completed_rps=17.0), _step(30, 1.0))) == 10
    assert sustained_rps((_step(10, 0.5), _step(20, 1.0, unexpected=19), _step(30, 1.0))) == 10
    assert sustained_rps((_step(10, 0.5), _step(20, 1.0, unexpected=18), _step(30, 1.0))) == 30
    assert sustained_rps((_step(10, 6.0), _step(20, 1.0))) == 0


def _record(seq: int, started_at: float, total_s: float) -> WireRecord:
    return WireRecord(
        seq=seq,
        marker=f"perfreq-{seq:06d}-n",
        caller="key-a1",
        endpoint="chat_completions",
        stream=False,
        blocked=False,
        started_at=started_at,
        status=200,
        header_s=total_s,
        first_byte_s=total_s,
        first_token_s=None,
        total_s=total_s,
        text="",
        ok=True,
        mock_delay_s=1.0,
        call_id=None,
        error=None,
    )


def test_a_floor_fails_when_the_second_step_s_latency_breaks_it_and_the_properties_carry_the_floor() -> None:
    ramp: Final = Ramp(first_rps=10.0, last_rps=20.0, step_rps=10.0, step_seconds=100.0, held_p99_seconds=2.0)
    case: Final = replace(
        CASE,
        traffic=ramped(CASE.traffic, ramp, seed=1, concurrency=50),
        limits=(Limit("sustained_rps", min=20),),
    )
    offsets: Final = send_offsets(ramp, seed=1)
    records: Final = tuple(
        _record(seq, 1000.0 + offset, 1.5 if ramp.step_of(seq) == 0 else 4.0)
        for seq, offset in enumerate(offsets, start=1)
    )
    observation: Final = Observation(
        records=records,
        mock_requests=(),
        guardrail_calls=(),
        rss_before=NO_RSS,
        rss_after=NO_RSS,
        wall_seconds=200.0,
        ramp=ramp,
    )
    first, second = observation.steps()
    assert first.held and first.gateway_added_p99 == 0.5 and first.completed_rps > 9.5
    assert not second.held and second.gateway_added_p99 == 3.0
    rows, exceeded = evaluate(case, observation)
    assert exceeded == ["sustained_rps: observed 10.0000 < floor 20.0000"]
    assert {("perf_sustained_rps", 10.0), ("perf_sustained_rps_floor", 20.0)} <= set(junit_properties(rows))
    assert dict(step_properties((first, second))) == {
        "perf_ramp_hold_p99_seconds": 2.0,
        "perf_ramp_step_10_gateway_added_p99": 0.5,
        "perf_ramp_step_10_completed_rps": first.completed_rps,
        "perf_ramp_step_10_unexpected": 0.0,
        "perf_ramp_step_20_gateway_added_p99": 3.0,
        "perf_ramp_step_20_completed_rps": second.completed_rps,
        "perf_ramp_step_20_unexpected": 0.0,
    }
    assert not evaluate(replace(case, limits=(Limit("sustained_rps", min=10),)), observation)[1]
