"""Tests for the coverage-registry tooling: pure logic plus a registry canary.

No `e2e` marker, so these run without a proxy. They exercise the coverage math and
the registry loader, and guard the checked-in registry against schema drift and
duplicate ids.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coverage_registry.collector import (
    compute_coverage,
    render,
    render_json,
    render_loki,
    render_prometheus,
)
from coverage_registry.registry import load_registry
from coverage_registry.schema import (
    GuardrailCell,
    LlmCell,
    LlmEndpoint,
    LoggingCell,
    Tier,
    loki_module_label,
)


def _llm(
    cell_id: str, tier: Tier, subject_endpoint: LlmEndpoint = "chat_completions"
) -> LlmCell:
    return LlmCell(
        id=cell_id,
        module="llm",
        tier=tier,
        assertions=("works",),
        source="test",
        subject_endpoint=subject_endpoint,
        route="openai",
        capability="basic",
        streaming="nonstream",
    )


def test_compute_coverage_counts_covered_p0_and_gaps() -> None:
    cells = (_llm("llm.a", Tier.P0), _llm("llm.b", Tier.P0), _llm("llm.c", Tier.P1))
    report = compute_coverage(cells, frozenset({"llm.a"}))
    assert (report.total, report.covered) == (3, 1)
    assert (report.p0_total, report.p0_covered) == (2, 1)
    assert report.p0_gaps == ("llm.b",)
    assert report.orphan_markers == ()


def test_orphan_marker_is_reported_not_counted() -> None:
    cells = (_llm("llm.a", Tier.P0),)
    report = compute_coverage(cells, frozenset({"llm.a", "llm.ghost"}))
    assert report.covered == 1
    assert report.orphan_markers == ("llm.ghost",)


def test_logging_and_guardrail_roll_up_into_one_module() -> None:
    cells = (
        LoggingCell(
            id="logging.x",
            module="logging",
            tier=Tier.P0,
            assertions=("logs_spend",),
            source="t",
            event="success",
            exercised_on=("chat_completions",),
        ),
        GuardrailCell(
            id="guardrail.y",
            module="guardrail",
            tier=Tier.P1,
            assertions=("blocks",),
            source="t",
            hook_point="pre_call",
            exercised_on=("chat_completions",),
        ),
    )
    report = compute_coverage(cells, frozenset())
    logging_and_guardrails = next(
        m for m in report.modules if m.module == "Logging & Guardrails"
    )
    assert logging_and_guardrails.total == 2


def test_llm_cells_roll_up_by_core_endpoint() -> None:
    cells = (
        _llm("llm.chat", Tier.P0, "chat_completions"),
        _llm("llm.messages", Tier.P0, "messages"),
        _llm("llm.responses", Tier.P1, "responses"),
        _llm("llm.batches", Tier.P0, "batches"),
        _llm("llm.realtime", Tier.P1, "realtime"),
    )
    report = compute_coverage(cells, frozenset({"llm.chat", "llm.batches"}))

    core = next(m for m in report.modules if m.module == "Core LLMs")
    non_core = next(m for m in report.modules if m.module == "Non-Core LLMs")

    assert (core.total, core.covered, core.p0_total, core.p0_covered) == (3, 1, 2, 1)
    assert (
        non_core.total,
        non_core.covered,
        non_core.p0_total,
        non_core.p0_covered,
    ) == (2, 1, 1, 1)


def test_text_render_uses_plain_coverage_language() -> None:
    report = compute_coverage(
        (_llm("llm.chat", Tier.P0), _llm("llm.batches", Tier.P0, "batches")),
        frozenset({"llm.chat"}),
    )

    text = render(report)

    assert "COVERAGE" in text
    assert "Headline coverage: 1/2  (50.0%)" in text
    assert "P0 COVERED" not in text


def test_json_render_exposes_module_coverage_for_grafana_jobs() -> None:
    report = compute_coverage(
        (_llm("llm.chat", Tier.P0), _llm("llm.batches", Tier.P0, "batches")),
        frozenset({"llm.chat"}),
    )

    payload = render_json(report)

    assert '"coverage_percent": 50.0' in payload
    assert '"module": "Core LLMs"' in payload
    assert '"module": "Non-Core LLMs"' in payload


def test_prometheus_render_exposes_module_coverage_timeseries() -> None:
    report = compute_coverage(
        (_llm("llm.chat", Tier.P0), _llm("llm.batches", Tier.P0, "batches")),
        frozenset({"llm.chat"}),
    )

    metrics = render_prometheus(report)

    assert 'litellm_e2e_coverage_cells{module="Core LLMs",state="covered"} 1' in metrics
    assert 'litellm_e2e_coverage_percent{module="Core LLMs"} 100.000000' in metrics
    assert 'litellm_e2e_coverage_percent{module="Non-Core LLMs"} 0.000000' in metrics
    assert "litellm_e2e_coverage_orphan_markers 0" in metrics


def test_loki_render_exposes_exact_stdout_lines_for_loki() -> None:
    report = compute_coverage(
        (_llm("llm.chat", Tier.P0), _llm("llm.batches", Tier.P0, "batches")),
        frozenset({"llm.chat"}),
    )

    lines = render_loki(report).splitlines()

    assert len(lines) == 1 + len(report.modules)
    assert lines[0] == "COVERAGE_TOTAL percent=50.0 covered=1 total=2"
    assert (
        lines[1] == "COVERAGE_MODULE module=core_llms percent=100.0 covered=1 total=1"
    )
    assert (
        lines[2] == "COVERAGE_MODULE module=non_core_llms percent=0.0 covered=0 total=1"
    )
    assert [line.split("module=", 1)[1].split(" ", 1)[0] for line in lines[1:]] == [
        loki_module_label(module.module) for module in report.modules
    ]
    assert all(
        " " not in line.split("module=", 1)[1].split(" ", 1)[0] for line in lines[1:]
    )


def test_real_registry_loads_and_ids_are_unique() -> None:
    cells = load_registry()
    ids = [c.id for c in cells]
    assert len(cells) > 250
    assert len(ids) == len(set(ids))
    assert any(c.id == "logging.prometheus.success.exports_metric" for c in cells)


def test_load_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = (
        "- {id: llm.dup, module: llm, tier: P0, assertions: [works], source: t, "
        "subject_endpoint: chat_completions, route: openai, capability: basic, streaming: nonstream}\n"
    )
    (tmp_path / "a.yaml").write_text(row)
    (tmp_path / "b.yaml").write_text(row)
    with pytest.raises(ValueError, match="duplicate cell ids"):
        load_registry(tmp_path)
