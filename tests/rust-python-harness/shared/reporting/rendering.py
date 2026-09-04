from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from typing_extensions import assert_never

from .models import CaseDisposition, CaseResult


@dataclass(frozen=True, slots=True)
class ReportSection:
    title: str
    blocks: tuple[str, ...]


class StrategyRenderer(Protocol):
    def __call__(self, results: Sequence[CaseResult]) -> tuple[ReportSection, ...]: ...


def render_case_outcome(result: CaseResult) -> str:
    prefix: Final = f"- {result.case.display_name}: {result.status.value}"
    spec: Final = result.case.spec
    match spec.disposition:
        case CaseDisposition.RUNNABLE:
            progress: Final = f", {len(result.completed)}/{result.total} checks" if result.total else ""
            return f"{prefix}{progress}, {spec.coverage.value} coverage"
        case CaseDisposition.NOT_IMPLEMENTED | CaseDisposition.SKIPPED:
            return f"{prefix}, {spec.reason}"
    assert_never(spec.disposition)
