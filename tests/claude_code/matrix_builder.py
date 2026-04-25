"""Matrix JSON Builder.

Pure-function module that consumes the pytest-produced `compat-results.json`,
the manifest, and run metadata, and emits the final `compatibility-matrix.json`
conforming to the schema published in the PRD.

This module is deliberately free of subprocess, network, or filesystem side
effects in its public API — the public entry points take pre-loaded inputs
and return data structures, so they can be exercised by golden-file tests
without I/O. A small `build_from_paths()` convenience wrapper does the I/O
for callers that need it (the daily-cron publisher).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

SCHEMA_VERSION = "1"
VALID_STATUSES = {"pass", "fail", "not_applicable", "not_tested"}


class ManifestError(ValueError):
    """Raised when `manifest.yaml` is malformed."""


class ResultsError(ValueError):
    """Raised when the pytest results artifact is malformed."""


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load and validate `manifest.yaml`.

    Returns a dict with keys: schema_version, providers, features. Raises
    ManifestError on missing fields or schema mismatch.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest at {path} is not a mapping")
    schema_version = str(raw.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema_version {schema_version!r} does not match "
            f"builder version {SCHEMA_VERSION!r}"
        )
    providers = raw.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ManifestError("manifest.providers must be a non-empty list")
    features = raw.get("features")
    if not isinstance(features, list) or not features:
        raise ManifestError("manifest.features must be a non-empty list")
    for feature in features:
        if not isinstance(feature, dict):
            raise ManifestError("each feature must be a mapping")
        if not feature.get("id") or not feature.get("name"):
            raise ManifestError("each feature must have id and name")
    return raw


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load the pytest results artifact and return its `results` list."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        raise ResultsError(f"results artifact at {path} has no `results` list")
    return raw["results"]


def build_matrix(
    *,
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    litellm_version: str,
    claude_code_version: str,
    generated_at: str,
) -> Dict[str, Any]:
    """Build the published matrix JSON from pre-loaded inputs.

    Empty cells (no test ran for a (feature, provider) and no
    `not_applicable` was declared) are filled in with `not_tested`. If
    multiple results report on the same cell — e.g. a per-feature test
    file containing one parametrize per Claude model — the cell aggregates
    to `pass` only if every model passed; otherwise `fail` with the first
    breaking model surfaced in the error.
    """
    providers: List[str] = list(manifest["providers"])
    feature_specs: List[Dict[str, Any]] = list(manifest["features"])

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for entry in results:
        if not isinstance(entry, Mapping):
            continue
        feature_id = entry.get("feature_id")
        provider = entry.get("provider")
        result = entry.get("result")
        if not feature_id or not provider or not isinstance(result, Mapping):
            continue
        if result.get("status") not in VALID_STATUSES:
            continue
        grouped.setdefault((feature_id, provider), []).append(dict(result))

    features_out: List[Dict[str, Any]] = []
    for spec in feature_specs:
        feature_id = spec["id"]
        cells: Dict[str, Dict[str, Any]] = {}
        for provider in providers:
            cell_results = grouped.get((feature_id, provider), [])
            cells[provider] = _aggregate_cell(cell_results)
        features_out.append(
            {
                "id": feature_id,
                "name": spec["name"],
                "providers": cells,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "litellm_version": litellm_version,
        "claude_code_version": claude_code_version,
        "providers": providers,
        "features": features_out,
    }


def _aggregate_cell(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of per-model results into a single cell status.

    Order of precedence (most informative wins):
      - Any `fail` → cell is `fail` with the first failure's error.
      - `not_applicable` → cell is `not_applicable` with the reason.
      - `pass` → cell is `pass`.
      - empty / nothing recognized → `not_tested`.
    """
    if not results:
        return {"status": "not_tested"}

    for r in results:
        if r.get("status") == "fail":
            return {"status": "fail", "error": str(r.get("error", "test failed"))}

    for r in results:
        if r.get("status") == "not_applicable":
            return {
                "status": "not_applicable",
                "reason": str(r.get("reason", "not applicable")),
            }

    if all(r.get("status") == "pass" for r in results):
        return {"status": "pass"}

    return {"status": "not_tested"}


def build_from_paths(
    *,
    manifest_path: Path,
    results_path: Path,
    litellm_version: str,
    claude_code_version: str,
    generated_at: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """I/O wrapper around build_matrix used by the publisher script."""
    manifest = load_manifest(manifest_path)
    results = load_results(results_path)
    matrix = build_matrix(
        manifest=manifest,
        results=results,
        litellm_version=litellm_version,
        claude_code_version=claude_code_version,
        generated_at=generated_at,
    )
    if output_path is not None:
        output_path.write_text(json.dumps(matrix, indent=2, sort_keys=False) + "\n")
    return matrix
