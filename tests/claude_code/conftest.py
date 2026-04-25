"""Pytest plumbing for the Claude Code compatibility matrix.

Two responsibilities live here:

1. The `compat_result` fixture — the only API a test author needs to learn.
   Tests call `compat_result.set({"status": "pass"})` (or fail / not_applicable)
   to report their outcome as a tagged union. The fixture is per-test and
   stores the last value reported.

2. The `pytest_runtest_makereport` hook — captures each test's reported result,
   infers (feature, provider) from the file path, and writes a single
   `compat-results.json` artifact next to JUnit XML. The Matrix JSON Builder
   consumes this artifact to produce the published `compatibility-matrix.json`.

The (feature, provider) inference comes from the test file path: the parent
directory name is the feature_id (matching `manifest.yaml`), and the file
stem after the leading `test_` is the provider id. This avoids per-file
metadata that drifts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

VALID_STATUSES = {"pass", "fail", "not_applicable", "not_tested"}
RESULTS_ARTIFACT_ENV = "COMPAT_RESULTS_PATH"
DEFAULT_ARTIFACT_PATH = "compat-results.json"


@dataclass
class CompatResult:
    """Per-test recorder for compatibility outcomes.

    Tests interact only via `.set(...)`. `.value` is read by the
    `pytest_runtest_makereport` hook after the test body finishes.
    """

    value: Optional[Dict[str, Any]] = None

    def set(self, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise TypeError("compat_result.set() requires a dict")
        status = result.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"compat_result.set() status must be one of {sorted(VALID_STATUSES)}, "
                f"got {status!r}"
            )
        if status == "fail" and not result.get("error"):
            raise ValueError("compat_result.set({'status': 'fail'}) requires 'error'")
        if status == "not_applicable" and not result.get("reason"):
            raise ValueError(
                "compat_result.set({'status': 'not_applicable'}) requires 'reason'"
            )
        self.value = dict(result)


@dataclass
class _CollectedResult:
    feature_id: str
    provider: str
    nodeid: str
    result: Dict[str, Any]


@dataclass
class _Collector:
    items: List[_CollectedResult] = field(default_factory=list)


_COLLECTOR = _Collector()


@pytest.fixture
def compat_result() -> CompatResult:
    """Per-test recorder for the (feature, provider) outcome.

    Tests should call `compat_result.set({"status": "pass"})` (or fail /
    not_applicable) before returning. If a test exits without calling `.set()`
    the harness records `status="fail"` with an explanatory error so that
    every collected node maps to a real cell.
    """
    return CompatResult()


def _infer_feature_and_provider(node_path: Path) -> Optional[tuple]:
    """Infer (feature_id, provider) from a test file path.

    Path shape:  tests/claude_code/<feature_id>/test_<provider>.py
    Returns None if the file is not a per-feature test (e.g. unit tests
    living under tests/claude_code/_driver_unit_tests/), so those don't
    pollute the matrix artifact.
    """
    name = node_path.name
    if not name.startswith("test_") or not name.endswith(".py"):
        return None
    provider = name[len("test_") : -len(".py")]
    feature_id = node_path.parent.name
    if feature_id.startswith("_") or feature_id == "claude_code":
        return None
    return feature_id, provider


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture compat_result.value at end-of-test and remember it for the artifact."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    inferred = _infer_feature_and_provider(Path(str(item.path)))
    if inferred is None:
        return
    feature_id, provider = inferred

    fixture = item.funcargs.get("compat_result") if hasattr(item, "funcargs") else None
    reported: Optional[Dict[str, Any]] = getattr(fixture, "value", None)

    if reported is None:
        if report.passed:
            reported = {
                "status": "fail",
                "error": "test passed without calling compat_result.set(); "
                "every compat test must report a status.",
            }
        else:
            reported = {
                "status": "fail",
                "error": (str(report.longrepr) if report.longrepr else "test failed"),
            }

    _COLLECTOR.items.append(
        _CollectedResult(
            feature_id=feature_id,
            provider=provider,
            nodeid=report.nodeid,
            result=reported,
        )
    )


def pytest_sessionfinish(session, exitstatus):
    """Write the structured results artifact at end of session."""
    artifact_path = os.environ.get(RESULTS_ARTIFACT_ENV) or DEFAULT_ARTIFACT_PATH
    payload = {
        "schema_version": "1",
        "results": [
            {
                "feature_id": item.feature_id,
                "provider": item.provider,
                "nodeid": item.nodeid,
                "result": item.result,
            }
            for item in _COLLECTOR.items
        ],
    }
    Path(artifact_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
