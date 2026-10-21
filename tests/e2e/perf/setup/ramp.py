"""A ramped run step by step: the send schedule its plan carries (`send_offsets`), one `StepReading` per step
and `sustained_rps`, the last step the pod held before the first it did not. `Ramp` itself sits in
`performance_test_case.py` with the rest of the case schema."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, takewhile
from typing import Final

from setup.performance_test_case import Ramp

HELD_SHARE: Final = 0.9
"""A step that completed fewer than this share of its rate is a queue that stopped draining."""
HELD_UNEXPECTED_SHARE: Final = 0.01
"""A step with more than this share of unexpected statuses broke, whatever its latency."""
_JITTER: Final = 0.25


def _step_offsets(ramp: Ramp, index: int, count: int, rng: random.Random) -> Iterator[float]:
    for slot in range(count):
        yield (index + (slot + 0.5 + rng.uniform(-_JITTER, _JITTER)) / count) * ramp.step_seconds


def send_offsets(ramp: Ramp, seed: int) -> tuple[float, ...]:
    """Seconds after the run starts to send each request of the plan, in plan order: a step's requests spread
    evenly over its window, each moved by up to a quarter of a slot either way, drawn from `seed` so a rerun
    sends at the same seconds."""
    rng: Final = random.Random(f"ramp-{seed}")
    return tuple(chain.from_iterable(_step_offsets(ramp, index, count, rng) for index, count in enumerate(ramp.counts)))


@dataclass(frozen=True, slots=True)
class StepReading:
    rps: float
    planned: int
    completed_rps: float
    gateway_added_p99: float | None
    unexpected: int
    held_p99_seconds: float
    gateway_added_p50: float | None = None
    time_to_first_token_p99: float | None = None
    cpu_seconds_per_request: float | None = None
    redis_latency_mean_seconds: float | None = None
    postgres_peak_connections: int | None = None
    """The hold verdict reads the first six. The rest say what the pod was doing on that step: CPU per request
    from the pod sampler, Redis latency from the gateway's `/metrics`, connections from `pg_stat_activity`,
    all cut to the step's window, so a latency jump can be told apart as CPU, waiting or neither."""

    @property
    def held(self) -> bool:
        return (
            self.gateway_added_p99 is not None
            and self.gateway_added_p99 <= self.held_p99_seconds
            and self.completed_rps >= HELD_SHARE * self.rps
            and self.unexpected <= HELD_UNEXPECTED_SHARE * self.planned
        )


def sustained_rps(steps: Iterable[StepReading]) -> float:
    """The rate of the last step that held before the first that did not, 0 when the first step did not hold."""
    held: Final = tuple(takewhile(lambda step: step.held, steps))
    return held[-1].rps if held else 0.0


def step_properties(steps: Iterable[StepReading]) -> tuple[tuple[str, float], ...]:
    """The per-step readings as JUnit properties (`perf_ramp_step_<rps>_<reading>`, every reading the step has)
    plus the hold bound, so the dashboard can draw the run's curves next to the sustained rate."""
    rows: Final = tuple(steps)
    hold: Final = (("perf_ramp_hold_p99_seconds", rows[0].held_p99_seconds),) if rows else ()
    return (*hold, *chain.from_iterable(_step_properties(step) for step in rows))


def _step_properties(step: StepReading) -> Iterator[tuple[str, float]]:
    prefix: Final = f"perf_ramp_step_{step.rps:g}"
    readings: Final[tuple[tuple[str, float | None], ...]] = (
        ("gateway_added_p99", step.gateway_added_p99),
        ("completed_rps", step.completed_rps),
        ("unexpected", float(step.unexpected)),
        ("gateway_added_p50", step.gateway_added_p50),
        ("time_to_first_token_p99", step.time_to_first_token_p99),
        ("cpu_seconds_per_request", step.cpu_seconds_per_request),
        ("redis_latency_mean_seconds", step.redis_latency_mean_seconds),
        (
            "postgres_peak_connections",
            None if step.postgres_peak_connections is None else float(step.postgres_peak_connections),
        ),
    )
    return ((f"{prefix}_{name}", value) for name, value in readings if value is not None)


def render_steps(steps: Iterable[StepReading]) -> tuple[str, ...]:
    rows: Final = tuple(steps)
    if not rows:
        return ()
    header: Final = (
        f"  steps: {'rps':>6} {'planned':>8} {'completed rps':>14} {'gateway added p50':>18} {'p99':>8}"
        f" {'ttft p99':>9} {'unexpected':>11} {'cpu s/req':>10} {'redis mean':>11} {'pg conns':>9}"
    )
    return (header, *(_step_line(step) for step in rows))


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _step_line(step: StepReading) -> str:
    conns: Final = "n/a" if step.postgres_peak_connections is None else str(step.postgres_peak_connections)
    return (
        f"         {step.rps:6.1f} {step.planned:8d} {step.completed_rps:14.2f} {_seconds(step.gateway_added_p50):>18}"
        f" {_seconds(step.gateway_added_p99):>8} {_seconds(step.time_to_first_token_p99):>9} {step.unexpected:11d}"
        f" {_seconds(step.cpu_seconds_per_request):>10} {_seconds(step.redis_latency_mean_seconds):>11} {conns:>9}"
        f"  {'held' if step.held else 'broke'}"
    )
