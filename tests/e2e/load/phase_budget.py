"""Comparing one load phase against another, for tests that degrade a dependency mid-run.

Two shapes of ceiling, because the metrics divide into two kinds. RSS and CPU are
machine-shaped: RSS scales with worker count and CPU with core count, so an absolute number
calibrated on one runner means nothing on the next, and what travels is the ratio against a
healthy phase measured on the same machine in the same run. Latency and log volume are not:
a ratio there is actively misleading, because a dependency that fails fast once its breaker
opens can make the degraded phase look cheaper than the healthy one while still being far
slower or noisier than a user should ever see. Those get a flat ceiling, which is the promise
the test is actually making.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias


def _rendered(value: float, unit: str, decimals: int) -> str:
    return f"{value:.{decimals}f}{unit}"


@dataclass(frozen=True, slots=True)
class RatioBudget:
    """One metric's healthy value, its degraded value, and how much growth is allowed."""

    name: str
    baseline: float
    degraded: float
    ratio_ceiling: float
    unit: str
    decimals: int = 1

    @property
    def ratio(self) -> float | None:
        """How many times the baseline the degraded value is, or None if there is no baseline."""
        return self.degraded / self.baseline if self.baseline > 0 else None

    def violation(self) -> str | None:
        """Why this metric fails its budget, or None if it passes."""
        ratio: Final = self.ratio
        if ratio is None:
            return (
                f"{self.name} measured {_rendered(self.baseline, self.unit, self.decimals)} in the healthy phase, "
                f"so there is nothing to compare the degraded phase against; the measurement did not happen"
            )
        if ratio > self.ratio_ceiling:
            return (
                f"{self.name} went from {_rendered(self.baseline, self.unit, self.decimals)} healthy to "
                f"{_rendered(self.degraded, self.unit, self.decimals)} degraded, {ratio:.1f}x the baseline and past "
                f"the {self.ratio_ceiling:.1f}x allowed"
            )
        return None


@dataclass(frozen=True, slots=True)
class AbsoluteBudget:
    """One metric's degraded value against a flat ceiling, for metrics a ratio cannot bound."""

    name: str
    measured: float
    ceiling: float
    unit: str
    decimals: int = 1

    def violation(self) -> str | None:
        """Why this metric fails its budget, or None if it passes."""
        if self.measured > self.ceiling:
            return (
                f"{self.name} measured {_rendered(self.measured, self.unit, self.decimals)} in the degraded phase, "
                f"past the {_rendered(self.ceiling, self.unit, self.decimals)} allowed"
            )
        return None


Budget: TypeAlias = RatioBudget | AbsoluteBudget


def violations(budgets: tuple[Budget, ...]) -> tuple[str, ...]:
    """Every budget the run blew, so one failure reports all of them instead of the first."""
    return tuple(violation for budget in budgets if (violation := budget.violation()) is not None)
