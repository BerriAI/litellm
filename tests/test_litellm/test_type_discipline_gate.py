"""Tests for scripts/type_discipline_gate.py.

The gate compares each LIT rule's codebase count against the merge-base count plus
the headroom the gate script grants, so what is pinned here is the identity that
keys those base counts, the headroom the gate deliberately keeps, and the diff scan
that turns a breach into file:line.
"""

import lint_base_counts
import type_discipline_gate as gate


def test_headroom_is_kept_only_for_the_seeded_final_and_rebind_rules():
    assert set(gate.HEADROOM) == {"LIT010", "LIT011"}
    assert all(isinstance(value, int) and value > 0 for value in gate.HEADROOM.values())


def test_the_checker_identity_is_keyed_on_the_checker_source():
    identity = gate.checker_identity()
    assert identity == lint_base_counts.Checker("type-discipline", (lint_base_counts.sha256_of(gate.CHECKER),))


def test_parse_changed_lines_groups_hunks_under_their_own_file():
    diff = (
        "diff --git a/litellm/a.py b/litellm/a.py\n"
        "--- a/litellm/a.py\n"
        "+++ b/litellm/a.py\n"
        "@@ -0,0 +3,2 @@\n"
        "+one\n"
        "+two\n"
        "diff --git a/litellm/b.py b/litellm/b.py\n"
        "--- a/litellm/b.py\n"
        "+++ b/litellm/b.py\n"
        "@@ -0,0 +10 @@\n"
        "+only\n"
    )
    changed = gate.parse_changed_lines(diff)
    assert changed["litellm/a.py"] == {3, 4}
    assert changed["litellm/b.py"] == {10}


def test_parse_changed_lines_handles_several_hunks_in_one_file():
    diff = (
        "+++ b/litellm/a.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+a\n"
        "@@ -9,0 +20,1 @@\n"
        "+b\n"
    )
    assert gate.parse_changed_lines(diff)["litellm/a.py"] == {1, 2, 20}


def test_parse_changed_lines_on_an_empty_diff_is_empty():
    assert gate.parse_changed_lines("") == {}


def test_introduced_keeps_only_violations_on_changed_lines():
    violations = (
        gate.Violation("litellm/a.py", 3, "LIT006"),
        gate.Violation("litellm/a.py", 99, "LIT006"),
        gate.Violation("litellm/b.py", 3, "LIT001"),
    )
    kept = gate.introduced(violations, {"litellm/a.py": {3}})
    assert kept == [gate.Violation("litellm/a.py", 3, "LIT006")]
