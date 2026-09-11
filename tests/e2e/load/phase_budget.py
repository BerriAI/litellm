"""Comparing one load phase against another, for tests that degrade a dependency mid-run.

A chaos phase's absolute numbers say very little on their own: RSS scales with worker count,
latency with core count, so a ceiling calibrated on one machine is meaningless on the next.
What travels is the ratio against a healthy phase measured on the same machine in the same run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Budget:
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

    def _rendered(self, value: float) -> str:
        return f"{value:.{self.decimals}f}{self.unit}"

    def violation(self) -> str | None:
        """Why this metric fails its budget, or None if it passes."""
        ratio: Final = self.ratio
        if ratio is None:
            return (
                f"{self.name} measured {self._rendered(self.baseline)} in the healthy phase, so there is nothing "
                f"to compare the degraded phase against; the measurement did not happen"
            )
        if ratio > self.ratio_ceiling:
            return (
                f"{self.name} went from {self._rendered(self.baseline)} healthy to "
                f"{self._rendered(self.degraded)} degraded, {ratio:.1f}x the baseline and past the "
                f"{self.ratio_ceiling:.1f}x allowed"
            )
        return None


def violations(budgets: tuple[Budget, ...]) -> tuple[str, ...]:
    """Every budget the run blew, so one failure reports all of them instead of the first."""
    return tuple(violation for budget in budgets if (violation := budget.violation()) is not None)
