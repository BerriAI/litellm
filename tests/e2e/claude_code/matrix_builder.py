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
      - Any `fail` → cell is `fail` with every failing model's error
        joined by `"; "` so a multi-tier breakage doesn't silently hide
        all but the first error from the published matrix.
      - Any `pass` → cell is `pass`. A mix of (pass, not_applicable) —
        e.g. a tier where the feature isn't supported alongside tiers
        where it works — surfaces as `pass` so the published cell
        reflects that the feature *does* work on this provider rather
        than silently demoting it to `not_applicable` and discarding
        the passing tiers.
      - All `not_applicable` → cell is `not_applicable` with the first
        row's reason.
      - empty / nothing recognized → `not_tested`.

    `not_tested` rows are treated as absent data: they're dropped before
    aggregation so a mix of (pass, not_tested) — e.g. from a partial
    crash or a test that explicitly recorded "this tier didn't run" —
    still surfaces the passing tiers rather than silently demoting the
    whole cell to `not_tested`. A cell is only `not_tested` when *every*
    row is `not_tested` (or there are no rows at all).
    """
    if not results:
        return {"status": "not_tested"}

    observed = [r for r in results if r.get("status") != "not_tested"]
    if not observed:
        return {"status": "not_tested"}

    failures = [r for r in observed if r.get("status") == "fail"]
    if failures:
        errors = [str(r.get("error", "test failed")) for r in failures]
        return {"status": "fail", "error": "; ".join(errors)}

    if any(r.get("status") == "pass" for r in observed):
        return {"status": "pass"}

    if all(r.get("status") == "not_applicable" for r in observed):
        return {
            "status": "not_applicable",
            "reason": str(observed[0].get("reason", "not applicable")),
        }

    return {"status": "not_tested"}


def _index_cells(matrix: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map ``(feature_id, provider) -> cell dict`` for a built matrix.

    Cells are keyed by the *stable* feature ``id`` (not the display
    ``name``, which can be reworded without changing the underlying row)
    and the provider key, so two matrices built at different times line up
    even if feature names drift.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for feature in matrix.get("features", []) or []:
        if not isinstance(feature, Mapping):
            continue
        feature_id = feature.get("id")
        if not feature_id:
            continue
        providers = feature.get("providers", {}) or {}
        if not isinstance(providers, Mapping):
            continue
        for provider, cell in providers.items():
            if isinstance(cell, Mapping):
                out[(feature_id, provider)] = dict(cell)
    return out


def find_regressions(
    old_matrix: Mapping[str, Any],
    new_matrix: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return the cells that flipped green→red (``pass`` → ``fail``).

    A *regression* is defined strictly: a cell that was ``pass`` in
    ``old_matrix`` and is ``fail`` in ``new_matrix``. Every other
    transition is intentionally *not* a regression:

      * ``red → green`` / ``green → green`` — the happy path.
      * ``red → red`` — a cell that is *already* failing for an unrelated
        reason (e.g. Anthropic out of API credits) must not block
        publishing, otherwise the daily PR would never auto-merge until
        that independent issue is fixed.
      * ``green → not_tested`` / ``green → not_applicable`` — a cell going
        grey is a degradation but not a *red* regression; treating a
        skipped/flaky run as a hard block would create false positives.

    Cells present only in ``new_matrix`` (a newly added feature or
    provider) have no baseline and therefore cannot be regressions.

    Each returned item is a flat str→str mapping so callers (the cron's
    ``check_regressions.py``) can render it without further lookups:
    ``feature_id``, ``feature_name``, ``provider``, ``old_status``,
    ``new_status``, ``error``.
    """
    old_cells = _index_cells(old_matrix)
    feature_names = {
        f.get("id"): str(f.get("name", f.get("id")))
        for f in new_matrix.get("features", []) or []
        if isinstance(f, Mapping) and f.get("id")
    }

    regressions: list[dict[str, str]] = []
    for (feature_id, provider), new_cell in sorted(
        _index_cells(new_matrix).items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        if new_cell.get("status") != "fail":
            continue
        old_cell = old_cells.get((feature_id, provider))
        if old_cell is None or old_cell.get("status") != "pass":
            continue
        regressions.append(
            {
                "feature_id": str(feature_id),
                "feature_name": feature_names.get(feature_id, str(feature_id)),
                "provider": str(provider),
                "old_status": "pass",
                "new_status": "fail",
                "error": str(new_cell.get("error", "")),
            }
        )
    return regressions


def build_from_paths(
    *,
    manifest_path: Path,
    results_path: Path,
    litellm_version: str,
    claude_code_version: str,
    generated_at: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """I/O wrapper around ``build_matrix``: reads the manifest and per-test results from disk, calls ``build_matrix``, and (optionally) writes the compat-matrix JSON to ``output_path``. Whatever orchestrator publishes the matrix (currently the ECR image) invokes this."""
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
