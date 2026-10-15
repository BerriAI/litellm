"""Tests for scripts/openapi_docs_gate.py.

The blame rule lives in scripts/lint_base_counts.py and is tested there; what is tested
here is the checker output parse and `introduced`, the diff scan that names the new
endpoint behind a breach.
"""

from pathlib import Path
from typing import Final

import openapi_docs_gate as gate


def test_editing_the_checker_rekeys_the_base_counts(tmp_path: Path) -> None:
    checker: Final = tmp_path / "check.py"
    checker.write_text("print('v1')\n")
    before: Final = gate.checker_identity(checker).artifact_name("abc123")
    checker.write_text("print('v2')\n")
    assert gate.checker_identity(checker).artifact_name("abc123") != before


def test_parse_violations_skips_anything_that_is_not_a_finding_line() -> None:
    out: Final = "Some warning from an import\nPOST /v1/decisions OA001\nPOST /v1/decisions OA003\nnot a line\n"
    assert gate.parse_violations(out) == (
        gate.Violation("POST", "/v1/decisions", "OA001"),
        gate.Violation("POST", "/v1/decisions", "OA003"),
    )


def test_count_by_rule_sums_findings_per_code() -> None:
    violations: Final = (
        gate.Violation("POST", "/a", "OA001"),
        gate.Violation("POST", "/b", "OA001"),
        gate.Violation("POST", "/b", "OA003"),
    )
    assert dict(gate.count_by_rule(violations)) == {"OA001": 2, "OA003": 1}


def test_introduced_keeps_only_routes_registered_on_added_lines() -> None:
    diff: Final = (
        "diff --git a/litellm/proxy/x.py b/litellm/proxy/x.py\n"
        "--- a/litellm/proxy/x.py\n"
        "+++ b/litellm/proxy/x.py\n"
        "@@ -0,0 +3,2 @@\n"
        '+@router.post("/v1/new")\n'
        "+async def new(request: Request): ...\n"
        "@@ -10,1 +20,1 @@\n"
        '-@router.post("/v1/removed")\n'
    )
    violations: Final = (
        gate.Violation("POST", "/v1/new", "OA001"),
        gate.Violation("POST", "/v1/old", "OA001"),
        gate.Violation("POST", "/v1/removed", "OA001"),
    )
    assert gate.introduced(violations, gate.added_text(diff)) == (gate.Violation("POST", "/v1/new", "OA001"),)


def test_introduced_does_not_match_a_path_that_is_only_a_prefix_of_an_added_one() -> None:
    added: Final = gate.added_text('+@router.post("/v1/decisions/batch")\n')
    assert gate.introduced((gate.Violation("POST", "/v1/decisions", "OA002"),), added) == ()
