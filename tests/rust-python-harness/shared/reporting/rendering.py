from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, assert_never

from .models import CaseDisposition, CaseResult


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
    lines: Final = tuple(_render_result(result) for result in results)
    return (ReportSection(label, lines),)


def _render_result(result: CaseResult) -> str:
    prefix: Final = f"- {result.case.surface}/{result.case.sdk_function}: {result.status.value}"
    spec: Final = result.case.spec
    match spec.disposition:
        case CaseDisposition.RUNNABLE:
            progress: Final = f", {len(result.completed)}/{result.total} checks" if result.total else ""
            return f"{prefix}{progress}, {spec.coverage.value} coverage"
        case CaseDisposition.NOT_IMPLEMENTED | CaseDisposition.SKIPPED:
            return f"{prefix}, {spec.reason}"
    assert_never(spec.disposition)
