"""Tests for marker-based e2e coverage collection."""

from __future__ import annotations

import pytest

from coverage_registry.check_coverage_sync import sync_errors
from coverage_registry.collector import (
    CoverageMarkers,
    compute_coverage,
    render_json,
    render_loki,
    render_prometheus,
)
from coverage_registry.schema import CoveragePoint

pytestmark = pytest.mark.e2e_coverage(
    module="other",
    endpoint="coverage_registry",
    provider="proxy",
    params=["collector", "sync_checker", "schema"],
)


def _markers() -> CoverageMarkers:
    return CoverageMarkers(
        points_by_nodeid={
            "test_core.py::test_chat": (
                CoveragePoint(
                    module="core_llms",
                    endpoint="/chat/completions",
                    provider="openai",
                    params=("tools", "streaming"),
                ),
            ),
            "test_budget.py::test_budget": (
                CoveragePoint(
                    module="budgets",
                    endpoint="/chat/completions",
                    provider="proxy",
                    params=("budget_enforcement",),
                ),
            ),
        },
        collected_nodeids=(
            "test_budget.py::test_budget",
            "test_core.py::test_chat",
            "test_unmarked.py::test_missing",
        ),
        unmarked_nodeids=("test_unmarked.py::test_missing",),
        invalid_markers=(),
        collection_errors=(),
    )


def test_compute_coverage_counts_unique_units_and_tests() -> None:
    report = compute_coverage(_markers())

    core = next(m for m in report.modules if m.module == "core_llms")
    budgets = next(m for m in report.modules if m.module == "budgets")

    assert report.total == 3
    assert report.test_count == 2
    assert report.collected_test_count == 3
    assert report.unmarked_test_count == 1
    assert (core.unit_count, core.test_count) == (2, 1)
    assert (budgets.unit_count, budgets.test_count) == (1, 1)


def test_json_render_exposes_marker_fields_for_grafana_jobs() -> None:
    payload = render_json(compute_coverage(_markers()))

    assert '"module": "core_llms"' in payload
    assert '"endpoint": "/chat/completions"' in payload
    assert '"provider": "openai"' in payload
    assert '"param": "tools"' in payload
    assert '"unmarked_test_count": 1' in payload


def test_prometheus_render_exposes_module_counts() -> None:
    metrics = render_prometheus(compute_coverage(_markers()))

    assert 'litellm_e2e_coverage_units{module="core_llms"} 2' in metrics
    assert 'litellm_e2e_coverage_tests{module="budgets"} 1' in metrics
    assert "litellm_e2e_coverage_unmarked_tests 1" in metrics
    assert "litellm_e2e_coverage_invalid_markers 0" in metrics


def test_loki_render_exposes_exact_stdout_lines_for_loki() -> None:
    report = compute_coverage(_markers())
    lines = render_loki(report).splitlines()

    assert len(lines) == 1 + len(report.modules)
    assert (
        lines[0] == "COVERAGE_TOTAL percent=100.0 covered=3 total=3 tests=2 "
        "unmarked_tests=1 invalid_markers=0"
    )
    assert lines[1] == (
        "COVERAGE_MODULE module=core_llms percent=100.0 covered=2 total=2 tests=1"
    )


def test_marker_schema_rejects_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="unknown endpoint"):
        CoveragePoint(
            module="core_llms",
            endpoint="/not-real",
            provider="openai",
            params=("tools",),
        )


def test_marker_schema_requires_params() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        CoveragePoint(
            module="core_llms",
            endpoint="/chat/completions",
            provider="openai",
            params=(),
        )


def test_collected_e2e_tests_have_coverage_metadata() -> None:
    assert sync_errors() == ()
