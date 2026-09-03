from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ...shared.reporting.models import CaseResult
from ...shared.reporting.rendering import ReportSection, render_case_outcome


def render_mapping_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    blocks: Final = tuple(render_case_outcome(result) for result in results)
    return (ReportSection("Python/Rust unit-test mappings", blocks or ("No mapping cases selected",)),)
