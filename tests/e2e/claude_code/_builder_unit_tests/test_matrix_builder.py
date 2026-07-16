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

from claude_code.json_types import JSON_OBJECT_ADAPTER, JSONValue
from claude_code.matrix_builder import (
    ManifestError,
    ResultsError,
    build_from_paths,
    build_matrix,
    load_manifest,
    load_results,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _as_dict(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def _as_list(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def _as_str(value: JSONValue) -> str:
    assert isinstance(value, str)
    return value


def test_build_matrix_matches_golden_file(tmp_path: Path) -> None:
    manifest = load_manifest(FIXTURES / "manifest.yaml")
    results = load_results(FIXTURES / "results.json")
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        generated_at="2026-04-25T00:00:00Z",
    )
    expected = JSON_OBJECT_ADAPTER.validate_python(json.loads((FIXTURES / "expected_matrix.json").read_text()))
    assert matrix == expected


def test_build_matrix_pass_requires_all_models_pass() -> None:
    """Multiple results in one cell must all be pass for the cell to be pass."""
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
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
    assert _as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"] == {"status": "pass"}


def test_build_matrix_any_fail_makes_cell_fail() -> None:
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
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
    cell = _as_dict(_as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"])
    assert cell["status"] == "fail"
    assert cell["error"] == "[claude-opus-4-7] timeout"


def test_build_matrix_joins_all_failure_errors_in_one_cell() -> None:
    """When multiple tiers fail for different reasons within the same cell,
    every failure's error must appear in the published cell so triage
    isn't reduced to a single tier's diagnostic.
    """
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "fail", "error": "[claude-haiku-4-5] 429"},
        },
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "fail", "error": "[claude-opus-4-7] timeout"},
        },
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    cell = _as_dict(_as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"])
    assert cell["status"] == "fail"
    assert "[claude-haiku-4-5] 429" in _as_str(cell["error"])
    assert "[claude-opus-4-7] timeout" in _as_str(cell["error"])


def test_build_matrix_mixed_pass_and_not_tested_surfaces_pass() -> None:
    """A `not_tested` row mixed with `pass` rows must not silently demote
    the cell to `not_tested` — `not_tested` is "absent data", not a
    negative signal. Otherwise a partial crash mid-test, or a test that
    explicitly recorded "tier didn't run", would discard real passing
    results from the published cell.
    """
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "not_tested"},
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
    assert _as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"] == {"status": "pass"}


def test_build_matrix_all_not_tested_stays_not_tested() -> None:
    """A cell whose every row is `not_tested` (or empty) must remain
    `not_tested` — the absent-data rule only drops `not_tested` rows
    when there's other signal to surface.
    """
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "not_tested"},
        },
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "not_tested"},
        },
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    assert _as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"] == {"status": "not_tested"}


def test_build_matrix_mixed_pass_and_not_applicable_surfaces_pass() -> None:
    """A `not_applicable` row mixed with `pass` rows must surface as
    `pass`, not `not_applicable`. The published cell answers "does this
    feature work on this provider?"; if any tier passes, the feature
    works there. Discarding passing tiers because one tier is NA would
    misrepresent the cell as unsupported.
    """
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {
                "status": "not_applicable",
                "reason": "haiku does not support extended thinking",
            },
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
    assert _as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"] == {"status": "pass"}


def test_build_matrix_all_not_applicable_stays_not_applicable() -> None:
    """When every observed row is `not_applicable`, the cell remains
    `not_applicable` and the first row's reason carries through to the
    published matrix.
    """
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {
                "status": "not_applicable",
                "reason": "feature unsupported on this provider",
            },
        },
        {
            "feature_id": "f",
            "provider": "anthropic",
            "result": {"status": "not_applicable", "reason": "ditto"},
        },
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    assert _as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["anthropic"] == {
        "status": "not_applicable",
        "reason": "feature unsupported on this provider",
    }


def test_build_matrix_fills_not_tested_for_missing_cells() -> None:
    manifest: dict[str, JSONValue] = {
        "schema_version": "1",
        "providers": ["anthropic", "azure"],
        "features": [{"id": "f", "name": "F"}],
    }
    results: list[JSONValue] = [
        {"feature_id": "f", "provider": "anthropic", "result": {"status": "pass"}},
    ]
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version="v",
        claude_code_version="c",
        generated_at="t",
    )
    cells = _as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])
    assert cells["anthropic"] == {"status": "pass"}
    assert cells["azure"] == {"status": "not_tested"}


def test_build_matrix_preserves_provider_and_feature_order() -> None:
    manifest: dict[str, JSONValue] = {
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
    assert [_as_dict(f)["id"] for f in _as_list(matrix["features"])] == ["z", "a"]
    assert list(_as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"]).keys()) == [
        "azure",
        "anthropic",
        "vertex_ai",
    ]


def test_build_matrix_emits_schema_version_one() -> None:
    manifest: dict[str, JSONValue] = {
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


def test_load_manifest_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        'schema_version: "2"\nproviders: [anthropic]\nfeatures:\n  - id: f\n    name: F\n'
    )
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(bad)


def test_load_manifest_rejects_empty_features(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text('schema_version: "1"\nproviders: [anthropic]\nfeatures: []\n')
    with pytest.raises(ManifestError):
        load_manifest(bad)


def test_load_results_rejects_missing_results_key(tmp_path: Path) -> None:
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps({"schema_version": "1"}))
    with pytest.raises(ResultsError):
        load_results(bad)


def test_build_matrix_6x5_grid_matches_published_sample() -> None:
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
        for feature in _as_list(full_manifest["features"])
        if _as_dict(feature)["id"] in v0_feature_ids
    ]
    manifest: dict[str, JSONValue] = {**full_manifest, "features": v0_features}

    feature_ids = [_as_str(_as_dict(feature)["id"]) for feature in _as_list(manifest["features"])]
    providers = [_as_str(provider) for provider in _as_list(manifest["providers"])]
    models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]

    results: list[JSONValue] = []
    for feature_id in feature_ids:
        for provider in providers:
            for model in models:
                results.append(
                    {
                        "feature_id": feature_id,
                        "provider": provider,
                        "nodeid": (
                            f"tests/e2e/claude_code/{feature_id}/test_{provider}.py"
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
    expected = JSON_OBJECT_ADAPTER.validate_python(
        json.loads((repo_root / "sample_compatibility-matrix.json").read_text())
    )
    assert matrix == expected


def test_build_matrix_1x5_grid_one_failing_model_breaks_cell() -> None:
    """If even one of three models fails on a provider, that cell is fail
    and the error string carries the failing model id so the docs
    tooltip can name the outlier."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(repo_root / "manifest.yaml")

    results: list[JSONValue] = [
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
    cell = _as_dict(_as_dict(_as_dict(_as_list(matrix["features"])[0])["providers"])["bedrock_invoke"])
    assert cell["status"] == "fail"
    assert "claude-opus-4-7-bedrock-invoke" in _as_str(cell["error"])


def test_build_from_paths_writes_output(tmp_path: Path) -> None:
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
    on_disk = JSON_OBJECT_ADAPTER.validate_python(json.loads(out.read_text()))
    assert on_disk == matrix
    expected = JSON_OBJECT_ADAPTER.validate_python(json.loads((FIXTURES / "expected_matrix.json").read_text()))
    assert on_disk == expected
