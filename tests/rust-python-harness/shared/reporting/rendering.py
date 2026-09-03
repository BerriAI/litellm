from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from .models import CaseResult


@dataclass(frozen=True, slots=True)
class ReportSection:
    title: str
    blocks: tuple[str, ...]


class StrategyRenderer(Protocol):
    def __call__(self, results: Sequence[CaseResult]) -> tuple[ReportSection, ...]: ...


def render_outcomes(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    if not results:
        return ()
    label: Final = results[0].case.strategy_label
    lines: Final = tuple(
        f"- {result.case.surface}/{result.case.sdk_function}: {result.status.value}"
        f"{f', {len(result.completed)}/{result.total} tests' if result.total else ''}, "
        f"{result.case.coverage.value} coverage"
        for result in results
    )
    return (ReportSection(label, lines),)
