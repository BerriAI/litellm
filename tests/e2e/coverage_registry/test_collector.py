"""Tests for the coverage-registry tooling: pure logic plus a registry canary.

No `e2e` marker, so these run without a proxy. They exercise the coverage math, the
id grammar, the product-surface generator, and the overlay loader, and guard the
generated-plus-overlay denominator against drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coverage_registry.collector import (
    compute_coverage,
    render,
    render_json,
    render_loki,
    render_prometheus,
)
from coverage_registry.overlay import load_overlay
from coverage_registry.product_surface import (
    ModelEntry,
    generate_llm_cell_ids,
)
from coverage_registry.registry import load_registry
from coverage_registry.schema import (
    Cell,
    Module,
    Tier,
    format_llm_id,
    loki_module_label,
    parse_llm_id,
)

_CHAT_BASIC = "llm.chat_completions.openai.basic.nonstream.works"
_MESSAGES_BASIC = "llm.messages.anthropic.basic.nonstream.works"
_RESPONSES_BASIC = "llm.responses.openai.basic.nonstream.works"
_BATCHES = "llm.batches.openai.create.nonstream.works"
_REALTIME = "llm.realtime.openai.session.na.works"


def _cell(cell_id: str, tier: Tier, module: Module = "llm") -> Cell:
    return Cell(id=cell_id, module=module, tier=tier)


def test_compute_coverage_counts_covered_p0_and_gaps() -> None:
    cells = (
        _cell(_CHAT_BASIC, Tier.P0),
        _cell(_MESSAGES_BASIC, Tier.P0),
        _cell(_RESPONSES_BASIC, Tier.P1),
    )
    report = compute_coverage(cells, frozenset({_CHAT_BASIC}))
    assert (report.total, report.covered) == (3, 1)
    assert (report.p0_total, report.p0_covered) == (2, 1)
    assert report.p0_gaps == (_MESSAGES_BASIC,)
    assert report.orphan_markers == ()


def test_orphan_marker_is_reported_not_counted() -> None:
    cells = (_cell(_CHAT_BASIC, Tier.P0),)
    report = compute_coverage(cells, frozenset({_CHAT_BASIC, "llm.ghost"}))
    assert report.covered == 1
    assert report.orphan_markers == ("llm.ghost",)


def test_logging_and_guardrail_roll_up_into_one_module() -> None:
    cells = (
        _cell("logging.langfuse.success.logs_spend", Tier.P0, "logging"),
        _cell("guardrail.presidio.pre_call.masks", Tier.P1, "guardrail"),
    )
    report = compute_coverage(cells, frozenset())
    logging_and_guardrails = next(m for m in report.modules if m.module == "Logging & Guardrails")
    assert logging_and_guardrails.total == 2


def test_llm_cells_roll_up_by_core_endpoint() -> None:
    cells = (
        _cell(_CHAT_BASIC, Tier.P0),
        _cell(_MESSAGES_BASIC, Tier.P0),
        _cell(_RESPONSES_BASIC, Tier.P1),
        _cell(_BATCHES, Tier.P0),
        _cell(_REALTIME, Tier.P1),
    )
    report = compute_coverage(cells, frozenset({_CHAT_BASIC, _BATCHES}))

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
        (_cell(_CHAT_BASIC, Tier.P0), _cell(_BATCHES, Tier.P0)),
        frozenset({_CHAT_BASIC}),
    )

    text = render(report)

    assert "COVERAGE" in text
    assert "Headline coverage: 1/2  (50.0%)" in text
    assert "P0 COVERED" not in text


def test_json_render_exposes_module_coverage_for_grafana_jobs() -> None:
    report = compute_coverage(
        (_cell(_CHAT_BASIC, Tier.P0), _cell(_BATCHES, Tier.P0)),
        frozenset({_CHAT_BASIC}),
    )

    payload = render_json(report)

    assert '"coverage_percent": 50.0' in payload
    assert '"module": "Core LLMs"' in payload
    assert '"module": "Non-Core LLMs"' in payload


def test_prometheus_render_exposes_module_coverage_timeseries() -> None:
    report = compute_coverage(
        (_cell(_CHAT_BASIC, Tier.P0), _cell(_BATCHES, Tier.P0)),
        frozenset({_CHAT_BASIC}),
    )

    metrics = render_prometheus(report)

    assert 'litellm_e2e_coverage_cells{module="Core LLMs",state="covered"} 1' in metrics
    assert 'litellm_e2e_coverage_percent{module="Core LLMs"} 100.000000' in metrics
    assert 'litellm_e2e_coverage_percent{module="Non-Core LLMs"} 0.000000' in metrics
    assert "litellm_e2e_coverage_orphan_markers 0" in metrics


def test_loki_render_exposes_exact_stdout_lines_for_loki() -> None:
    report = compute_coverage(
        (_cell(_CHAT_BASIC, Tier.P0), _cell(_BATCHES, Tier.P0)),
        frozenset({_CHAT_BASIC}),
    )

    lines = render_loki(report).splitlines()

    assert len(lines) == 1 + len(report.modules)
    assert lines[0] == "COVERAGE_TOTAL percent=50.0 covered=1 total=2"
    assert lines[1] == "COVERAGE_MODULE module=core_llms percent=100.0 covered=1 total=1"
    assert lines[2] == "COVERAGE_MODULE module=non_core_llms percent=0.0 covered=0 total=1"
    assert [line.split("module=", 1)[1].split(" ", 1)[0] for line in lines[1:]] == [
        loki_module_label(module.module) for module in report.modules
    ]
    assert all(" " not in line.split("module=", 1)[1].split(" ", 1)[0] for line in lines[1:])


def test_parse_llm_id_round_trips_and_rejects_non_vocab() -> None:
    parsed = parse_llm_id(_CHAT_BASIC)
    assert parsed is not None
    assert (parsed.endpoint, parsed.route, parsed.capability, parsed.streaming) == (
        "chat_completions",
        "openai",
        "basic",
        "nonstream",
    )
    assert (
        format_llm_id(
            parsed.endpoint,
            parsed.route,
            parsed.capability,
            parsed.streaming,
            parsed.assertion,
        )
        == _CHAT_BASIC
    )
    assert parse_llm_id("llm.chat_completions.openai.not_a_capability.nonstream.works") is None
    assert parse_llm_id("mgmt.key.generate.persists") is None


def test_generator_gates_flagged_capabilities_on_model_metadata() -> None:
    entries = (
        ModelEntry(litellm_provider="openai", mode="chat", supports_function_calling=True),
        ModelEntry(litellm_provider="anthropic", mode="chat"),
    )
    generated = generate_llm_cell_ids(entries)

    assert "llm.chat_completions.openai.basic.nonstream.works" in generated
    assert "llm.chat_completions.anthropic.basic.nonstream.works" in generated
    assert "llm.chat_completions.openai.tool_use.nonstream.works" in generated
    assert "llm.chat_completions.anthropic.tool_use.nonstream.works" not in generated


def test_generator_leaves_messages_matrix_ungated() -> None:
    entries = (ModelEntry(litellm_provider="anthropic", mode="chat"),)
    generated = generate_llm_cell_ids(entries)

    assert "llm.messages.anthropic.tool_use.nonstream.works" in generated
    assert "llm.messages.anthropic.web_search.nonstream.works" in generated
    assert "llm.messages.anthropic.service_tier.nonstream.works" not in generated


def test_real_registry_is_generated_superset_of_overlay_and_ids_are_unique() -> None:
    cells = load_registry()
    ids = tuple(c.id for c in cells)
    assert len(cells) > 250
    assert len(ids) == len(set(ids))

    denominator = frozenset(ids)
    assert generate_llm_cell_ids() <= denominator
    assert frozenset(load_overlay()) <= denominator
    assert "logging.prometheus.success.exports_metric" in denominator


def test_generated_cell_absent_from_overlay_defaults_to_p2(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("logging.custom.success.logs_spend: {tier: P0}\n")
    cells = load_registry(overlay)
    by_id = {c.id: c for c in cells}

    curated = by_id["logging.custom.success.logs_spend"]
    assert curated.tier is Tier.P0

    generated_id = "llm.chat_completions.openai.basic.nonstream.works"
    assert by_id[generated_id].tier is Tier.P2


def test_load_registry_rejects_unknown_module_prefix(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("nope.some.cell: {tier: P0}\n")
    with pytest.raises(ValidationError):
        load_registry(overlay)
