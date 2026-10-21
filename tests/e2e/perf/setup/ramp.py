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
    """The per-step readings as JUnit properties, `perf_ramp_step_<rps>_gateway_added_p99` and
    `perf_ramp_step_<rps>_completed_rps`, so the dashboard can draw the run's curve next to the sustained rate."""
    return tuple(chain.from_iterable(_step_property_pair(step) for step in steps))


def _step_property_pair(step: StepReading) -> tuple[tuple[str, float], ...]:
    prefix: Final = f"perf_ramp_step_{step.rps:g}"
    p99: Final = () if step.gateway_added_p99 is None else ((f"{prefix}_gateway_added_p99", step.gateway_added_p99),)
    return (*p99, (f"{prefix}_completed_rps", step.completed_rps))


def render_steps(steps: Iterable[StepReading]) -> tuple[str, ...]:
    rows: Final = tuple(steps)
    if not rows:
        return ()
    header: Final = (
        f"  steps: {'rps':>6} {'planned':>8} {'completed rps':>14} {'gateway added p99':>18} {'unexpected':>11}"
    )
    return (header, *(_step_line(step) for step in rows))


def _step_line(step: StepReading) -> str:
    p99: Final = "n/a" if step.gateway_added_p99 is None else f"{step.gateway_added_p99:.4f}"
    return (
        f"         {step.rps:6.1f} {step.planned:8d} {step.completed_rps:14.2f} {p99:>18} {step.unexpected:11d}"
        f"  {'held' if step.held else 'broke'}"
    )
