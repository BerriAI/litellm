"""Pytest plumbing for the 4h-harness suites.

These suites hold the live provider tests migrated out of the per-commit
CircleCI jobs (`llm_translation_testing`, `llm_responses_api_testing`) by the
chat-scope keep/drop audit. The harness owns scheduling, deploy, credentials,
rate limiting, result merging, and alerting; suite authors only write tests.

The moved tests still import shared bases and helpers from their original
directories (`base_llm_unit_tests`, `base_responses_api`, sibling test
modules), so both directories are put on sys.path here.

`compat_result` mirrors the tagged-union recorder contract from
tests/claude_code/conftest.py (the compat matrix, suite #1 on the harness).
This is a stub: it validates and stores results but the merge-to-
`compat-results.json` hook belongs to the harness-engineer's contract and is
not duplicated here. Tests that do not call it are reported by pass/fail as
usual.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "tests", "llm_translation"),
    os.path.join(_REPO_ROOT, "tests", "llm_responses_api_testing"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VALID_STATUSES = {"pass", "fail", "not_applicable", "not_tested"}


@dataclass
class CompatResult:
    value: Optional[Dict[str, Any]] = None
    values: List[Dict[str, Any]] = field(default_factory=list)

    def set(self, result: Dict[str, Any]) -> None:
        self.value = self._validate(result)

    def add(self, result: Dict[str, Any]) -> None:
        self.values.append(self._validate(result))

    @staticmethod
    def _validate(result: Dict[str, Any]) -> Dict[str, Any]:
        status = result.get("status") if isinstance(result, dict) else None
        if status not in VALID_STATUSES:
            raise ValueError(
                f"compat_result status must be one of {sorted(VALID_STATUSES)}, got {status!r}"
            )
        return result


@pytest.fixture
def compat_result() -> CompatResult:
    return CompatResult()
