"""Tests for the coverage-registry tooling: pure logic plus a registry canary.

No `e2e` marker, so these run without a proxy. They exercise the coverage math and
the registry loader, and guard the checked-in registry against schema drift and
duplicate ids.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coverage_registry.collector import compute_coverage
from coverage_registry.registry import load_registry
from coverage_registry.schema import GuardrailCell, LlmCell, LoggingCell, Tier


def _llm(cell_id: str, tier: Tier) -> LlmCell:
    return LlmCell(
        id=cell_id,
        module="llm",
        tier=tier,
        assertions=("works",),
        source="test",
        subject_endpoint="chat_completions",
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
    logging_and_guardrails = next(m for m in report.modules if m.module == "Logging & Guardrails")
    assert logging_and_guardrails.total == 2


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
