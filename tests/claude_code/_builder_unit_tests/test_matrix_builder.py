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
