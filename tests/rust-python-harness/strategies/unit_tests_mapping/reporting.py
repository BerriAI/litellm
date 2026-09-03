from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ...shared.reporting.models import CaseResult
from ...shared.reporting.rendering import ReportSection, render_case_outcome
from .runner import MAPPING_REPORT_ARTIFACT


def _render_result(result: CaseResult) -> str:
    reports: Final = tuple(
        artifact.body
        for artifacts in result.artifacts.values()
        for artifact in artifacts
        if artifact.kind == MAPPING_REPORT_ARTIFACT
    )
    return "\n".join((render_case_outcome(result), *reports))


def render_mapping_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    blocks: Final = tuple(_render_result(result) for result in results)
    return (ReportSection("Python/Rust unit-test mappings", blocks or ("No mapping cases selected",)),)
