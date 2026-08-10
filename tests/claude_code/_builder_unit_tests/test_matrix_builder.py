"""Golden-file tests for the Matrix JSON Builder.

These tests fix the published JSON schema. The builder is a pure function
from (manifest, results, metadata) → matrix dict, so we feed it a fixture
input set and compare the produced dict to a checked-in expected output.

Any schema drift — intentional or accidental — surfaces as a diff in PR
review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.claude_code.matrix_builder import (
    ManifestError,
    ResultsError,
    build_from_paths,
    build_matrix,
    find_regressions,
    load_manifest,
    load_results,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_matrix_matches_golden_file(tmp_path):
    manifest = load_manifest(FIXTURES / "manifest.yaml")
    results = load_results(FIXTURES / "results.json")
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        generated_at="2026-04-25T00:00:00Z",
    )
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text())
    assert matrix == expected


def test_build_matrix_pass_requires_all_models_pass():
    """Multiple results in one cell must all be pass for the cell to be pass."""
    manifest = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results = [
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    assert matrix["features"][0]["providers"]["anthropic"] == {"status": "pass"}


def test_build_matrix_any_fail_makes_cell_fail():
    manifest = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results = [
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "fail", "error": "[claude-opus-4-7] timeout"},
        },
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    cell = matrix["features"][0]["providers"]["anthropic"]
    assert cell["status"] == "fail"
    assert cell["error"] == "[claude-opus-4-7] timeout"


def _single_cell_matrix(results):
    """Build a 1x1 matrix and return its only cell. Helper for the
    not_applicable aggregation tests below."""
    manifest = {
        "schema_version": "1",
        "providers": ["vertex_ai"],
        "features": [{"id": "f", "name": "F"}],
    }
    matrix = build_matrix(
        manifest=manifest,
        results=[
            {"feature_id": "f", "provider": "vertex_ai", "result": r} for r in results
        ],
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    return matrix["features"][0]["providers"]["vertex_ai"]


def test_build_matrix_not_applicable_is_neutral_when_others_pass():
    """A tier that doesn't support the feature (`not_applicable`) must not
    drag down a cell whose supported tiers all pass. Models the real
    Vertex AI count_tokens case: Haiku 4.5 is unsupported, Sonnet/Opus
    pass → the cell is green."""
    cell = _single_cell_matrix(
        [
            {"status": "not_applicable", "reason": "[haiku] not supported"},
            {"status": "pass"},
            {"status": "pass"},
        ]
    )
    assert cell == {"status": "pass"}


def test_build_matrix_all_not_applicable_makes_cell_not_applicable():
    """If *every* tier is not_applicable, the cell is not_applicable and
    surfaces the first reason."""
    cell = _single_cell_matrix(
        [
            {"status": "not_applicable", "reason": "first reason"},
            {"status": "not_applicable", "reason": "second reason"},
        ]
    )
    assert cell == {"status": "not_applicable", "reason": "first reason"}


def test_build_matrix_fail_beats_not_applicable():
    """A genuine failure still reds the cell even when another tier is
    not_applicable — failures win over the neutral skip."""
    cell = _single_cell_matrix(
        [
            {"status": "not_applicable", "reason": "[haiku] not supported"},
            {"status": "fail", "error": "[opus] regression"},
        ]
    )
    assert cell["status"] == "fail"
    assert cell["error"] == "[opus] regression"


def test_build_matrix_fills_not_tested_for_missing_cells():
    manifest = {
        "schema_version": "1",
        "providers": ["anthropic", "azure"],
        "features": [{"id": "f", "name": "F"}],
    }
    results = [
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    cells = matrix["features"][0]["providers"]
    assert cells["anthropic"] == {"status": "pass"}
    assert cells["azure"] == {"status": "not_tested"}


def test_build_matrix_preserves_provider_and_feature_order():
    manifest = {
        "schema_version": "1",
        "providers": ["azure", "anthropic", "vertex_ai"],
        "features": [
            {"id": "z", "name": "Z"},
            {"id": "a", "name": "A"},
        ],
    }
    matrix = build_matrix(
        manifest=manifest,
        results=[],
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    assert matrix["providers"] == ["azure", "anthropic", "vertex_ai"]
    assert [f["id"] for f in matrix["features"]] == ["z", "a"]
    assert list(matrix["features"][0]["providers"].keys()) == [
        "azure",
        "anthropic",
        "vertex_ai",
    ]


def test_build_matrix_emits_schema_version_one():
    manifest = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    matrix = build_matrix(
        manifest=manifest,
        results=[],
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    assert matrix["schema_version"] == "1"


def test_load_manifest_rejects_wrong_schema_version(tmp_path):
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        'schema_version: "2"\nproviders: [anthropic]\nfeatures:\n  - id: f\n    name: F\n'
    )
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(bad)


def test_load_manifest_rejects_empty_features(tmp_path):
    bad = tmp_path / "manifest.yaml"
    bad.write_text('schema_version: "1"\nproviders: [anthropic]\nfeatures: []\n')
    with pytest.raises(ManifestError):
        load_manifest(bad)


def test_load_results_rejects_missing_results_key(tmp_path):
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps({"schema_version": "1"}))
    with pytest.raises(ResultsError):
        load_results(bad)


def test_build_matrix_6x5_grid_matches_published_sample():
    """Slice 5 acceptance: feeding the per-model results the full v0
    row set produces reproduces the hand-authored 6x5 sample that the
    docs page renders.

    Inputs mirror the structure of `compat-results.json` after a real
    run with the proxy configured for all five columns and all six
    feature directories: every (feature, provider, model) cell yields a
    `pass`. Anthropic announced Claude in Microsoft Foundry on
    2025-11-18, so the Azure column is now exercised end-to-end like
    the others rather than reporting `not_applicable`.

    The aggregated matrix must equal the checked-in
    `sample_compatibility-matrix.json` byte-for-byte (after JSON load),
    so any future schema drift surfaces here in review.
    """
    repo_root = Path(__file__).resolve().parents[1]
    full_manifest = load_manifest(repo_root / "manifest.yaml")

    # The v0 sample matrix is a frozen baseline: it covers exactly the
    # six features the PRD shipped with, in their canonical order. The
    # live manifest may carry additional rows (extensions added after
    # v0 shipped), but the sample is derived only from the v0 slice so
    # this test stays a meaningful regression gate for the v0 cell
    # shape rather than chasing every new row added downstream.
    v0_feature_ids = [
        "basic_messaging_non_streaming",
        "basic_messaging_streaming",
        "tool_use",
        "prompt_caching_5m",
        "vision",
        # Row 6 of the v0 PRD; originally shipped as `extended_thinking`.
        # The id was renamed in-place to `thinking` to match Anthropic's
        # current docs (which reserve "extended thinking" for the
        # deprecated manual mode only). The row's *position* in v0 is
        # the load-bearing invariant, not the id string.
        "thinking",
    ]
    v0_features = [
        feature
        for feature in full_manifest["features"]
        if feature["id"] in v0_feature_ids
    ]
    manifest = {**full_manifest, "features": v0_features}

    feature_ids = [feature["id"] for feature in manifest["features"]]
    providers = manifest["providers"]
    models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]

    results = []
    for feature_id in feature_ids:
        for provider in providers:
            for model in models:
                results.append(
                    {
                        "feature_id": feature_id,
                        "provider": provider,
                        "nodeid": (
                            f"tests/claude_code/{feature_id}/test_{provider}.py"
                            f"::test[{model}]"
                        ),
                        "result": {"status": "pass"},
                    }
                )

    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        generated_at="2026-04-25T00:00:00Z",
    )
    expected = json.loads((repo_root / "sample_compatibility-matrix.json").read_text())
    assert matrix == expected


def test_build_matrix_1x5_grid_one_failing_model_breaks_cell():
    """If even one of three models fails on a provider, that cell is fail
    and the error string carries the failing model id so the docs
    tooltip can name the outlier."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(repo_root / "manifest.yaml")

    results = [
        {
            "feature_id": "basic_messaging_non_streaming",
            "provider": "bedrock_invoke",
            "result": {"status": "pass"},
        },
        {
            "feature_id": "basic_messaging_non_streaming",
            "provider": "bedrock_invoke",
            "result": {
                "status": "fail",
                "error": "[claude-opus-4-7-bedrock-invoke] claude CLI exited 1: throttled",
            },
        },
        {
            "feature_id": "basic_messaging_non_streaming",
            "provider": "bedrock_invoke",
            "result": {"status": "pass"},
        },
    ]

    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    cell = matrix["features"][0]["providers"]["bedrock_invoke"]
    assert cell["status"] == "fail"
    assert "claude-opus-4-7-bedrock-invoke" in cell["error"]


def _matrix(cells, *, names=None):
    """Build a minimal matrix dict from a {(feature_id, provider): status}
    or {(feature_id, provider): cell_dict} mapping. Helper for the
    find_regressions tests below."""
    names = names or {}
    features = {}
    for (feature_id, provider), value in cells.items():
        cell = {"status": value} if isinstance(value, str) else dict(value)
        features.setdefault(feature_id, {})[provider] = cell
    return {
        "features": [
            {
                "id": feature_id,
                "name": names.get(feature_id, feature_id.upper()),
                "providers": providers,
            }
            for feature_id, providers in features.items()
        ]
    }


def test_find_regressions_flags_pass_to_fail():
    old = _matrix({("vision", "anthropic"): "pass"})
    new = _matrix(
        {("vision", "anthropic"): {"status": "fail", "error": "credit balance too low"}}
    )
    regressions = find_regressions(old, new)
    assert len(regressions) == 1
    r = regressions[0]
    assert r["feature_id"] == "vision"
    assert r["provider"] == "anthropic"
    assert r["old_status"] == "pass"
    assert r["new_status"] == "fail"
    assert r["error"] == "credit balance too low"


def test_find_regressions_ignores_red_to_red():
    """An already-failing cell that stays failing is NOT a regression — a
    provider that's independently broken (e.g. out of credits) must not
    block the daily auto-merge forever."""
    old = _matrix({("vision", "anthropic"): "fail"})
    new = _matrix({("vision", "anthropic"): "fail"})
    assert find_regressions(old, new) == []


def test_find_regressions_ignores_improvements_and_steady_green():
    old = _matrix(
        {
            ("vision", "anthropic"): "fail",  # red -> green
            ("tool_use", "azure"): "pass",  # green -> green
        }
    )
    new = _matrix(
        {
            ("vision", "anthropic"): "pass",
            ("tool_use", "azure"): "pass",
        }
    )
    assert find_regressions(old, new) == []


def test_find_regressions_ignores_green_to_grey():
    """green→not_tested / green→not_applicable are degradations but not
    *red* regressions; we deliberately don't block on them."""
    old = _matrix(
        {
            ("vision", "azure"): "pass",
            ("tool_use", "azure"): "pass",
        }
    )
    new = _matrix(
        {
            ("vision", "azure"): "not_tested",
            ("tool_use", "azure"): {"status": "not_applicable", "reason": "skip"},
        }
    )
    assert find_regressions(old, new) == []


def test_find_regressions_ignores_new_cells_without_baseline():
    """A cell only present in the new matrix (new feature/provider) has no
    baseline, so a fail there can't be a regression."""
    old = _matrix({("vision", "anthropic"): "pass"})
    new = _matrix(
        {
            ("vision", "anthropic"): "pass",
            ("brand_new_feature", "anthropic"): "fail",
        }
    )
    assert find_regressions(old, new) == []


def test_find_regressions_matches_by_id_not_name():
    """Renaming a feature's display name must not hide a regression: cells
    are matched on the stable id."""
    old = _matrix({("thinking", "anthropic"): "pass"}, names={"thinking": "Old Name"})
    new = _matrix(
        {("thinking", "anthropic"): "fail"}, names={"thinking": "Totally New Name"}
    )
    regressions = find_regressions(old, new)
    assert len(regressions) == 1
    assert regressions[0]["feature_id"] == "thinking"
    assert regressions[0]["feature_name"] == "Totally New Name"


def test_find_regressions_reports_multiple_sorted():
    old = _matrix(
        {
            ("vision", "anthropic"): "pass",
            ("tool_use", "anthropic"): "pass",
            ("vision", "azure"): "pass",
        }
    )
    new = _matrix(
        {
            ("vision", "anthropic"): "fail",
            ("tool_use", "anthropic"): "fail",
            ("vision", "azure"): "pass",  # stays green
        }
    )
    regressions = find_regressions(old, new)
    keys = [(r["feature_id"], r["provider"]) for r in regressions]
    assert keys == [("tool_use", "anthropic"), ("vision", "anthropic")]


def test_find_regressions_empty_old_matrix_is_safe():
    """No baseline at all (first publish) yields no regressions."""
    new = _matrix({("vision", "anthropic"): "fail"})
    assert find_regressions({}, new) == []


def test_build_from_paths_writes_output(tmp_path):
    out = tmp_path / "compatibility-matrix.json"
    matrix = build_from_paths(
        manifest_path=FIXTURES / "manifest.yaml",
        results_path=FIXTURES / "results.json",
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        generated_at="2026-04-25T00:00:00Z",
        output_path=out,
    )
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == matrix
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text())
    assert on_disk == expected
