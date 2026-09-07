from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ...shared.reporting.models import CaseResult
from ...shared.reporting.rendering import ReportSection, render_case_outcome


def render_rust_unit_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    blocks: Final = tuple(render_case_outcome(result) for result in results)
    return (ReportSection("Native Rust unit-test outcomes", blocks or ("No Rust unit-test cases selected",)),)
