"""Unit tests for `find_regressions`, the green→red detector that gates
auto-merge on the daily compat-matrix docs PR (see `cron_vm/`).

Markerless harness tests: they exercise publisher plumbing, not a product
feature, so they run without a proxy and carry no `e2e` marker.
"""

from __future__ import annotations

from typing import Mapping, Union

from claude_code.matrix_builder import find_regressions

_CellSpec = Union[str, Mapping[str, str]]


def _matrix(
    cells: Mapping[tuple[str, str], _CellSpec],
    *,
    names: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a minimal matrix dict from a {(feature_id, provider): status}
    or {(feature_id, provider): cell_dict} mapping."""
    names = names or {}
    features: dict[str, dict[str, dict[str, str]]] = {}
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


def test_find_regressions_flags_pass_to_fail() -> None:
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


def test_find_regressions_ignores_red_to_red() -> None:
    """An already-failing cell that stays failing is NOT a regression — a
    provider that's independently broken (e.g. out of credits) must not
    block the daily auto-merge forever."""
    old = _matrix({("vision", "anthropic"): "fail"})
    new = _matrix({("vision", "anthropic"): "fail"})
    assert find_regressions(old, new) == []


def test_find_regressions_ignores_improvements_and_steady_green() -> None:
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


def test_find_regressions_ignores_green_to_grey() -> None:
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


def test_find_regressions_ignores_new_cells_without_baseline() -> None:
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


def test_find_regressions_matches_by_id_not_name() -> None:
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


def test_find_regressions_reports_multiple_sorted() -> None:
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


def test_find_regressions_empty_old_matrix_is_safe() -> None:
    """No baseline at all (first publish) yields no regressions."""
    new = _matrix({("vision", "anthropic"): "fail"})
    assert find_regressions({}, new) == []
