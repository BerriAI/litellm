from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pydantic import ValidationError

from ...shared.reporting.models import CaseResult
from ...shared.reporting.rendering import ReportSection, render_case_outcome
from .mapping_report import MappingReportArtifact, mapping_report_lines
from .runner import MAPPING_REPORT_ARTIFACT


def _render_artifact(body: str) -> str:
    try:
        artifact: Final = MappingReportArtifact.model_validate_json(body)
    except ValidationError as error:
        return f"Mapping report artifact is invalid: {error}"
    return "\n".join(mapping_report_lines(artifact.report, detailed=artifact.detailed))


def _render_result(result: CaseResult) -> str:
    reports: Final = tuple(
        _render_artifact(artifact.body)
        for artifacts in result.artifacts.values()
        for artifact in artifacts
        if artifact.kind == MAPPING_REPORT_ARTIFACT
    )
    if reports:
        return "\n".join((f"Case: {result.case.display_name}", *reports))
    return render_case_outcome(result)


def render_mapping_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    blocks: Final = tuple(_render_result(result) for result in results)
    return (ReportSection("Python/Rust unit-test mappings", blocks or ("No mapping cases selected",)),)
