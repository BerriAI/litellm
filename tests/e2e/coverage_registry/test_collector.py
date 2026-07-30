"""Tests for the coverage-registry tooling: pure logic plus a registry canary.

No `e2e` marker, so these run without a proxy. They exercise the coverage math and
the registry loader, and guard the checked-in registry against schema drift and
duplicate ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coverage_registry.collector import (
    UI_DECLARATION_FILE,
    CliArgs,
    compute_coverage,
    covered_ids,
    exit_code,
    load_ui_declarations,
    render,
    render_json,
    render_loki,
    render_prometheus,
    scan_covers_markers,
)
from coverage_registry.registry import load_registry
from coverage_registry.schema import (
    GuardrailCell,
    LlmCell,
    LlmEndpoint,
    LoggingCell,
    MgmtCell,
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


def test_registry_has_no_anthropic_embeddings_row() -> None:
    """Anthropic ships no embeddings API and litellm has no handler for one, so a
    row asserting it can never pass. An unreachable row inflates the denominator
    forever."""
    llm_cells = tuple(c for c in load_registry() if isinstance(c, LlmCell))
    assert not [
        c
        for c in llm_cells
        if c.subject_endpoint == "embeddings" and c.route == "anthropic"
    ]


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _absent(tmp_path: Path) -> Path:
    return tmp_path / "no-ui-declaration.yaml"


def test_marker_on_a_deselected_test_still_counts(tmp_path: Path) -> None:
    """A cell is covered when a test declaring it exists. `tests/e2e/load/conftest.py`
    deselects the weekly anomaly test unless an env var is set, and a conftest that
    drops items must not be able to move the coverage number."""
    _write(
        tmp_path / "conftest.py",
        "import pytest\n\n\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    dropped = [i for i in items if i.get_closest_marker('gated') is not None]\n"
        "    config.hook.pytest_deselected(items=dropped)\n"
        "    items[:] = [i for i in items if i.get_closest_marker('gated') is None]\n",
    )
    _write(
        tmp_path / "test_gated.py",
        "import pytest\n\n\n"
        "@pytest.mark.gated\n"
        "@pytest.mark.covers('llm.gated.cell')\n"
        "def test_gated() -> None:\n"
        "    pass\n",
    )

    result = covered_ids(tmp_path, _absent(tmp_path))
    ids, errors = result.ids, result.collection_errors

    assert "llm.gated.cell" in ids
    assert errors == ()


def test_marker_behind_an_optional_dependency_guard_still_counts(tmp_path: Path) -> None:
    """`tests/e2e/mcp/test_mcp_chat_completion_oauth_e2e.py` sits behind
    `pytest.importorskip`, so on a runner without the optional dep the module is
    never imported and pytest reports no items for it. Its cells still exist."""
    _write(
        tmp_path / "test_optional.py",
        "import pytest\n\n"
        "pytest.importorskip('a_package_that_is_not_installed_anywhere')\n\n\n"
        "@pytest.mark.covers('mcp.optional.cell')\n"
        "def test_optional() -> None:\n"
        "    pass\n",
    )

    result = covered_ids(tmp_path, _absent(tmp_path))
    ids, errors = result.ids, result.collection_errors

    assert "mcp.optional.cell" in ids
    assert errors == ()


def test_markers_built_at_import_time_still_count(tmp_path: Path) -> None:
    """`batches/test_batches_e2e.py` computes its ids from a helper, so the source
    text alone cannot see them; the collect-only pytest pass has to stay."""
    _write(
        tmp_path / "test_dynamic.py",
        "import pytest\n\n\n"
        "def cells() -> tuple[str, ...]:\n"
        "    return ('llm.' + 'dynamic.cell',)\n\n\n"
        "@pytest.mark.parametrize(\n"
        "    'case', [pytest.param('a', marks=pytest.mark.covers(*cells()))]\n"
        ")\n"
        "def test_dynamic(case: str) -> None:\n"
        "    pass\n",
    )

    assert "llm.dynamic.cell" in covered_ids(tmp_path, _absent(tmp_path)).ids


def test_module_that_cannot_be_imported_is_reported_as_a_collection_error(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "test_broken.py",
        "import a_package_that_is_not_installed_anywhere  # noqa: F401\n",
    )

    errors = covered_ids(tmp_path, _absent(tmp_path)).collection_errors

    assert any("test_broken.py" in error for error in errors)


def test_unparseable_source_is_reported_rather_than_silently_dropped(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "test_syntax.py", "def test_x(:\n")

    ids, errors = scan_covers_markers(tmp_path)

    assert ids == frozenset()
    assert any("test_syntax.py" in error for error in errors)


_KEYS_SPEC = """import { test, expect } from "@playwright/test";

test.describe("Proxy Admin - Keys", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Update key TPM and RPM limits", async ({ page }) => {
    await page.getByRole("button", { name: "Save Changes" }).click();
  });

  test.skip(!process.env.LITELLM_LICENSE, "proxy is running unlicensed");
});
"""

_UI_CELL = MgmtCell(
    id="mgmt.key.update.happy_path",
    module="mgmt",
    tier=Tier.P0,
    assertions=("happy_path",),
    source="t",
    surface="ui",
)


def _ui_suite(tmp_path: Path, *, spec: str = _KEYS_SPEC, title: str) -> Path:
    """A miniature ui/ suite: one spec plus a declaration claiming `title` in it."""
    _write(tmp_path / "ui" / "tests" / "proxy-admin" / "keys.spec.ts", spec)
    declaration = tmp_path / "ui" / "coverage.yaml"
    _write(
        declaration,
        "covers:\n"
        "  - id: mgmt.key.update.happy_path\n"
        "    spec: tests/proxy-admin/keys.spec.ts\n"
        f"    test: {json.dumps(title)}\n",
    )
    return declaration


def test_ui_declaration_feeds_the_covered_set(tmp_path: Path) -> None:
    """The TypeScript Playwright suite emits no pytest markers, so its cells reach
    the numerator through a checked-in declaration file instead."""
    declaration = _ui_suite(tmp_path, title="Update key TPM and RPM limits")

    declared = load_ui_declarations(declaration)

    assert declared.ids == frozenset({"mgmt.key.update.happy_path"})
    assert declared.unresolved == ()
    assert compute_coverage((_UI_CELL,), covered_ids(tmp_path, declaration).ids).covered == 1


def test_ui_declaration_stops_counting_when_its_test_is_renamed(
    tmp_path: Path,
) -> None:
    """Renaming or deleting the Playwright test must drop the cell out of the
    numerator, so a claim here can never outlive the test that backs it."""
    declaration = _ui_suite(tmp_path, title="Update key TPM and RPM limits (renamed)")

    declared = load_ui_declarations(declaration)

    assert declared.ids == frozenset()
    assert declared.unresolved == (
        "mgmt.key.update.happy_path: tests/proxy-admin/keys.spec.ts has no test "
        "titled 'Update key TPM and RPM limits (renamed)'",
    )
    report = compute_coverage(
        (_UI_CELL,), declared.ids, stale_ui_declarations=declared.unresolved
    )
    assert report.covered == 0
    assert "no test titled" in render(report)


def test_ui_declaration_stops_counting_when_its_spec_is_deleted(
    tmp_path: Path,
) -> None:
    declaration = tmp_path / "ui" / "coverage.yaml"
    _write(
        declaration,
        "covers:\n"
        "  - id: mgmt.key.update.happy_path\n"
        "    spec: tests/proxy-admin/keys.spec.ts\n"
        "    test: Update key TPM and RPM limits\n",
    )

    declared = load_ui_declarations(declaration)

    assert declared.ids == frozenset()
    assert declared.unresolved == (
        "mgmt.key.update.happy_path: spec tests/proxy-admin/keys.spec.ts does not exist",
    )


_SPEC_WITH_DYNAMIC_SIBLING = """import { test } from "@playwright/test";

test(`${segment}: sidebar nav and reload`, async ({ page }) => {});

test("Update key rate limits", async ({ page }) => {});
"""


def test_a_dynamic_sibling_title_does_not_exempt_the_rest_of_the_spec(
    tmp_path: Path,
) -> None:
    """The exemption is per declaration, never per file. A renamed literal test
    must still fail even when an interpolated title sits beside it in the same
    spec, otherwise one templated title switches validation off for the file."""
    declaration = _ui_suite(
        tmp_path,
        spec=_SPEC_WITH_DYNAMIC_SIBLING,
        title="Update key TPM and RPM limits",
    )

    declared = load_ui_declarations(declaration)

    assert declared.ids == frozenset()
    assert declared.unresolved == (
        "mgmt.key.update.happy_path: tests/proxy-admin/keys.spec.ts has no test "
        "titled 'Update key TPM and RPM limits'",
    )


def test_an_interpolated_title_is_matched_by_its_literal_segments(
    tmp_path: Path,
) -> None:
    """An interpolated title is still checked as far as it can be: the declared
    title has to be one the template could actually have produced."""
    declaration = _ui_suite(
        tmp_path,
        spec=_SPEC_WITH_DYNAMIC_SIBLING,
        title="api-keys: sidebar nav and reload",
    )

    declared = load_ui_declarations(declaration)

    assert declared.ids == frozenset({"mgmt.key.update.happy_path"})
    assert declared.unresolved == ()


def test_a_title_the_template_cannot_produce_is_rejected(tmp_path: Path) -> None:
    declaration = _ui_suite(
        tmp_path,
        spec=_SPEC_WITH_DYNAMIC_SIBLING,
        title="api-keys: some other flow entirely",
    )

    assert load_ui_declarations(declaration).ids == frozenset()


def test_a_wholly_interpolated_title_backs_no_declaration(tmp_path: Path) -> None:
    """A title that is nothing but interpolation would match anything, so it is
    not allowed to satisfy a row; the contributor has to give the test a title
    with something literal in it."""
    spec = (
        'import { test } from "@playwright/test";\n\n'
        "test(`${title}`, async ({ page }) => {});\n"
    )
    declaration = _ui_suite(tmp_path, spec=spec, title="Update key TPM and RPM limits")

    assert load_ui_declarations(declaration).ids == frozenset()


def test_a_commented_out_test_backs_no_declaration(tmp_path: Path) -> None:
    """Commenting a test out while debugging and forgetting to restore it leaves
    the title sitting in the source. It must not keep the cell counted."""
    spec = (
        'import { test } from "@playwright/test";\n\n'
        '// test("Update key TPM and RPM limits", async ({ page }) => {});\n'
    )
    declaration = _ui_suite(tmp_path, spec=spec, title="Update key TPM and RPM limits")

    assert load_ui_declarations(declaration).ids == frozenset()


def test_a_block_commented_test_backs_no_declaration(tmp_path: Path) -> None:
    spec = (
        'import { test } from "@playwright/test";\n\n'
        "/*\n"
        'test("Update key TPM and RPM limits", async ({ page }) => {});\n'
        "*/\n"
    )
    declaration = _ui_suite(tmp_path, spec=spec, title="Update key TPM and RPM limits")

    assert load_ui_declarations(declaration).ids == frozenset()


def test_a_title_containing_a_url_still_validates(tmp_path: Path) -> None:
    """Stripping comments must not touch `//` inside a string. A false negative
    here would be worse than the commented-out case it guards against."""
    title = "redirects to https://example.com/ui after login"
    spec = (
        'import { test } from "@playwright/test";\n\n'
        f'test("{title}", async ({{ page }}) => {{}});\n'
    )
    declaration = _ui_suite(tmp_path, spec=spec, title=title)

    declared = load_ui_declarations(declaration)

    assert declared.ids == frozenset({"mgmt.key.update.happy_path"})
    assert declared.unresolved == ()


def test_a_comment_does_not_swallow_the_titles_after_it(tmp_path: Path) -> None:
    """An apostrophe in a comment must not open a string that eats the rest of the
    file; the real specs are full of comments like this one."""
    spec = (
        'import { test } from "@playwright/test";\n\n'
        "// antd's dropdown renders off-viewport during the open animation\n"
        "/* the modal's footer button text varies between versions */\n"
        'test("Update key TPM and RPM limits", async ({ page }) => {});\n'
    )
    declaration = _ui_suite(tmp_path, spec=spec, title="Update key TPM and RPM limits")

    assert load_ui_declarations(declaration).ids == frozenset(
        {"mgmt.key.update.happy_path"}
    )


def test_a_title_assembled_from_variables_backs_no_declaration(
    tmp_path: Path,
) -> None:
    spec = (
        'import { test } from "@playwright/test";\n\n'
        "test(TITLE, async ({ page }) => {});\n"
    )
    declaration = _ui_suite(tmp_path, spec=spec, title="Update key TPM and RPM limits")

    assert load_ui_declarations(declaration).ids == frozenset()


def test_stale_ui_declaration_fails_strict(tmp_path: Path) -> None:
    """A dead declaration and an unknown cell id are the same failure: the tree no
    longer supports what the registry claims. Both must fail --strict."""
    declaration = _ui_suite(tmp_path, title="Update key TPM and RPM limits (renamed)")
    declared = load_ui_declarations(declaration)
    report = compute_coverage(
        (_UI_CELL,), declared.ids, stale_ui_declarations=declared.unresolved
    )
    strict = CliArgs(format="text", strict=True, fail_on_collection_errors=False)
    lenient = CliArgs(format="text", strict=False, fail_on_collection_errors=False)

    assert exit_code(report, strict) == 1
    assert exit_code(report, lenient) == 0


def test_ui_declaration_id_outside_the_registry_is_an_orphan(tmp_path: Path) -> None:
    declaration = _ui_suite(tmp_path, title="Update key TPM and RPM limits")
    _write(
        declaration,
        "covers:\n"
        "  - id: mgmt.key.typo.happy_path\n"
        "    spec: tests/proxy-admin/keys.spec.ts\n"
        "    test: Update key TPM and RPM limits\n",
    )

    report = compute_coverage(
        (_llm("llm.a", Tier.P0),), load_ui_declarations(declaration).ids
    )

    assert report.covered == 0
    assert report.orphan_markers == ("mgmt.key.typo.happy_path",)


def test_missing_ui_declaration_claims_nothing(tmp_path: Path) -> None:
    declared = load_ui_declarations(_absent(tmp_path))
    assert (declared.ids, declared.unresolved) == (frozenset(), ())


def test_checked_in_ui_declaration_resolves_and_names_real_registry_cells() -> None:
    registry_ids = frozenset(c.id for c in load_registry())
    declared = load_ui_declarations(UI_DECLARATION_FILE)
    assert declared.unresolved == ()
    assert declared.ids <= registry_ids
