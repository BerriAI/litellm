import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "ruff_strict_gate.py"
_spec = importlib.util.spec_from_file_location("ruff_strict_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

Violation = gate.Violation

_ENABLED_BY_RUFF_DEFAULTS = frozenset({"F401"})


def rule(name, limit):
    return {name: {"limit": limit}}


def test_under_ceiling_passes():
    assert gate.evaluate({"ANN001": 100}, {"ANN001": 100}, rule("ANN001", 110)) == []


def test_ceiling_is_the_limit_boundary():
    budget = rule("ANN001", 110)
    at = gate.evaluate({"ANN001": 110}, {"ANN001": 90}, budget)
    over = gate.evaluate({"ANN001": 111}, {"ANN001": 90}, budget)
    assert at == []
    assert [b.rule for b in over] == ["ANN001"]
    assert over[0].cap == 110
    assert over[0].added == 21


def test_over_ceiling_and_change_added_fails():
    breaches = gate.evaluate({"C901": 11}, {"C901": 9}, rule("C901", 10))
    assert [b.rule for b in breaches] == ["C901"]
    assert breaches[0].added == 2


def test_base_already_over_ceiling_change_added_nothing_is_not_blamed():
    # drift safety: base is over limit, this change leaves the count where it is
    assert gate.evaluate({"C901": 15}, {"C901": 15}, rule("C901", 10)) == []


def test_change_that_reduces_an_over_ceiling_rule_is_not_blamed():
    # still over limit, but moving the right direction
    assert gate.evaluate({"C901": 14}, {"C901": 16}, rule("C901", 10)) == []


def test_rules_are_independent():
    budget = {**rule("ANN001", 150), **rule("C901", 10)}
    breaches = gate.evaluate(
        {"ANN001": 130, "C901": 11}, {"ANN001": 100, "C901": 10}, budget
    )
    assert [b.rule for b in breaches] == ["C901"]  # ANN001 130 <= 150, C901 11 > 10


def test_missing_rule_counts_as_zero():
    assert gate.evaluate({}, {}, rule("C901", 0)) == []


def test_update_ratchets_limit_down_by_what_the_branch_fixed_never_up():
    budget = {**rule("ANN001", 150), **rule("C901", 10)}
    # ANN001 fixed 20 (100 -> 80) so its limit falls 150 -> 130; C901 grew, so its
    # limit holds flat at 10 (a fix must never loosen a ceiling).
    current = {"ANN001": 80, "C901": 12}
    base = {"ANN001": 100, "C901": 9}
    assert gate.ratcheted_budget(budget, current, base) == {
        "ANN001": {"limit": 130},
        "C901": {"limit": 10},
    }


def test_parse_changed_lines_maps_added_lines_per_file():
    diff = (
        "+++ b/litellm/a.py\n"
        "@@ -10 +10,3 @@\n+x\n+y\n+z\n"
        "+++ b/litellm/b.py\n"
        "@@ -5,2 +7 @@\n+q\n"
    )
    changed = gate.parse_changed_lines(diff)
    assert changed["litellm/a.py"] == {10, 11, 12}
    assert changed["litellm/b.py"] == {7}


def test_introduced_keeps_only_violations_on_changed_lines():
    violations = [
        Violation("litellm/a.py", 10, "ANN001"),
        Violation("litellm/a.py", 99, "C901"),
    ]
    assert gate.introduced(violations, {"litellm/a.py": {10}}) == [
        Violation("litellm/a.py", 10, "ANN001")
    ]


@pytest.mark.parametrize("hunk", ["@@ -1 +1 @@", "@@ -1,0 +1,2 @@"])
def test_parse_changed_lines_handles_single_and_ranged_hunks(hunk):
    assert gate.parse_changed_lines(f"+++ b/litellm/a.py\n{hunk}\n")["litellm/a.py"]


def test_over_ceiling_flags_only_counts_above_the_limit():
    budget = rule("C901", 10)
    assert gate.over_ceiling({"C901": 10}, budget) == frozenset()
    assert gate.over_ceiling({"C901": 11}, budget) == frozenset({"C901"})
    assert gate.over_ceiling({}, budget) == frozenset()


def test_over_ceiling_ignores_rules_missing_from_the_budget():
    assert gate.over_ceiling({"NEW99": 100}, rule("C901", 10)) == frozenset()


def test_over_ceiling_is_independent_across_rules():
    budget = {**rule("ANN001", 150), **rule("C901", 10)}
    assert gate.over_ceiling({"ANN001": 130, "C901": 11}, budget) == frozenset({"C901"})


def _git(cwd, *args):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(cwd, name):
    (cwd / name).write_text(name)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "-m", name)
    return _git(cwd, "rev-parse", "HEAD")


def _branched_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate@example.com")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    branch_point = _commit(repo, "shared.txt")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.txt")
    _git(repo, "checkout", "-q", "main")
    base_tip = _commit(repo, "drift.txt")
    _git(repo, "checkout", "-q", "feature")
    return repo, branch_point, base_tip


def test_base_point_is_the_branch_point_when_no_merge_is_in_progress(tmp_path):
    repo, branch_point, _ = _branched_repo(tmp_path)
    assert gate.resolve_base_point("main", cwd=repo) == branch_point


def test_base_point_mid_merge_advances_to_the_merged_in_base_tip(tmp_path):
    repo, _, base_tip = _branched_repo(tmp_path)
    _git(repo, "merge", "--no-commit", "--no-ff", "main")
    assert gate.resolve_base_point("main", cwd=repo) == base_tip


def _lint_section(config_name: str) -> dict:
    return tomllib.loads((_REPO_ROOT / config_name).read_text())["lint"]


def _base_external() -> tuple[str, ...]:
    return tuple(_lint_section("ruff.toml")["external"])


def _strict_external() -> tuple[str, ...]:
    return tuple(_lint_section("ruff-strict.toml")["external"])


def _strict_selected() -> frozenset:
    return frozenset(_lint_section("ruff-strict.toml")["select"])


def _prefix_covered(code: str, prefixes: tuple[str, ...]) -> bool:
    return any(code.startswith(prefix) for prefix in prefixes)


def _selected_by_the_normal_config() -> frozenset:
    return frozenset(_lint_section("ruff.toml")["extend-select"]) | _ENABLED_BY_RUFF_DEFAULTS


def _budgeted_rules() -> frozenset:
    return frozenset(json.loads((_REPO_ROOT / "ruff-strict-budget.json").read_text()))


def _ruff_binary() -> str | None:
    beside_interpreter = Path(sys.executable).with_name("ruff")
    return str(beside_interpreter) if beside_interpreter.exists() else shutil.which("ruff")


_RUFF = _ruff_binary()
_needs_ruff = pytest.mark.skipif(_RUFF is None, reason="ruff is not installed in this environment")


def _ruff_output_for_noqa(code: str, *extra_args: str) -> str:
    proc = subprocess.run(
        [
            _RUFF,
            "check",
            "--no-cache",
            "--stdin-filename",
            "litellm/types/_external_probe.py",
            *extra_args,
            "-",
        ],
        cwd=_REPO_ROOT,
        input=f"def _probe(x: int):  # noqa: {code}\n    return x\n",
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_every_strict_gate_rule_is_protected_from_base_ruf100():
    unprotected = frozenset(
        selector
        for selector in _strict_selected()
        if not _prefix_covered(selector, _base_external())
        and selector not in _selected_by_the_normal_config()
    )
    assert unprotected == frozenset(), (
        f"`ruff check` deletes any `# noqa` naming {sorted(unprotected)} as unused, so suppressing "
        "one of those strict-gate rules breaks lint. Cover them in ruff.toml's lint.external or "
        "enable them in its lint.extend-select."
    )


def test_every_selected_rule_keeps_stale_noqa_detection_somewhere():
    policed_by_strict = frozenset(
        selector
        for selector in _strict_selected()
        if not _prefix_covered(selector, _strict_external())
    )
    policed_by_base = frozenset(
        selector
        for selector in _selected_by_the_normal_config()
        if not _prefix_covered(selector, _base_external())
    )
    shadowed = (
        _strict_selected() | _selected_by_the_normal_config()
    ) - policed_by_strict - policed_by_base
    assert shadowed == frozenset(), (
        f"no config's RUF100 can ever report a stale `# noqa` for {sorted(shadowed)}: every config "
        "that selects each of them also shadows it with an external entry. Narrow the external "
        "entry in ruff.toml or ruff-strict.toml."
    )


_BASE_OWNED_FAMILY = re.compile(r"E[479]\d+|F\d+|T20\d+")
_BASE_OWNED_SINGLES = frozenset({"PGH004", "RUF008", "RUF009", "RUF100"})


@pytest.fixture(scope="module")
def all_ruff_rule_codes() -> frozenset:
    listing = subprocess.run(
        [_RUFF, "rule", "--all", "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    assert listing.returncode == 0, listing.stderr
    return frozenset(
        entry["code"] for entry in json.loads(listing.stdout) if "Removed" not in entry["status"]
    )


@_needs_ruff
def test_every_base_owned_rule_is_external_or_selected_in_the_strict_config(all_ruff_rule_codes):
    base_owned = frozenset(
        code
        for code in all_ruff_rule_codes
        if _BASE_OWNED_FAMILY.fullmatch(code) or code in _BASE_OWNED_SINGLES
    )
    stranded = frozenset(
        code
        for code in base_owned
        if code not in _strict_selected() and not _prefix_covered(code, _strict_external())
    )
    assert stranded == frozenset(), (
        f"the strict gate's RUF100 reads a valid `# noqa` for {sorted(stranded)} as unused, the "
        "spurious-breach trap ruff-strict.toml's external override exists to prevent. Cover them "
        "there."
    )
    double_booked = frozenset(
        code
        for code in base_owned
        if code in _strict_selected() and _prefix_covered(code, _strict_external())
    )
    assert double_booked == frozenset(), (
        f"{sorted(double_booked)} are selected by the strict config yet shadowed by its external "
        "list, so their stale suppressions can never be reported. Narrow the external entry in "
        "ruff-strict.toml."
    )


def test_every_budgeted_rule_is_one_the_gate_actually_measures():
    selectors = tuple(_lint_section("ruff-strict.toml")["select"])
    unmeasured = frozenset(code for code in _budgeted_rules() if not code.startswith(selectors))
    assert unmeasured == frozenset(), (
        f"the gate never counts {sorted(unmeasured)}, so their ceilings are dead config that reads "
        "as coverage. Either select them in ruff-strict.toml or drop them from the budget."
    )


@_needs_ruff
def test_every_strict_selected_rule_is_budgeted_or_hard_failed_by_the_base_config(all_ruff_rule_codes):
    strict_enabled = frozenset(
        code
        for code in all_ruff_rule_codes
        if code.startswith(tuple(_lint_section("ruff-strict.toml")["select"]))
    )
    base_hard_failed = tuple(_lint_section("ruff.toml")["extend-select"])
    unpoliced = frozenset(
        code
        for code in strict_enabled
        if code not in _budgeted_rules()
        and not code.startswith(base_hard_failed)
        and code not in _ENABLED_BY_RUFF_DEFAULTS
    )
    assert unpoliced == frozenset(), (
        f"nothing enforces {sorted(unpoliced)}: the gate skips rules missing from the budget, and "
        "the base config does not hard-fail them. Re-add a budget ceiling or graduate them into "
        "ruff.toml's lint.extend-select."
    )


@_needs_ruff
def test_a_noqa_for_a_strict_gate_rule_survives_the_normal_ruff_run():
    assert "RUF100" not in _ruff_output_for_noqa("ANN202")


@_needs_ruff
def test_the_external_list_is_what_saves_that_noqa():
    assert "RUF100" in _ruff_output_for_noqa("ANN202", "--config", "lint.external=[]")


@_needs_ruff
def test_a_stale_noqa_for_a_locally_enabled_rule_is_still_reported():
    assert "RUF100" in _ruff_output_for_noqa("F401")


def _ruff_output_for_source(source: str) -> str:
    proc = subprocess.run(
        [_RUFF, "check", "--no-cache", "--stdin-filename", "litellm/types/_graduate_probe.py", "-"],
        cwd=_REPO_ROOT,
        input=source,
        capture_output=True,
        text=True,
    )
    return proc.stdout


_DEPRECATED_TYPING_ALIAS = "from typing import List  # noqa: UP035\n\n\ndef _probe(x: List[int]) -> None: ...\n"


@_needs_ruff
def test_a_graduated_rule_now_fails_the_normal_ruff_run_instead_of_waiting_for_the_gate():
    assert "UP006" in _ruff_output_for_source(_DEPRECATED_TYPING_ALIAS)


@_needs_ruff
def test_a_graduated_rule_can_still_be_suppressed_without_tripping_unused_noqa():
    suppressed = _DEPRECATED_TYPING_ALIAS.replace("...\n", "...  # noqa: UP006\n")
    output = _ruff_output_for_source(suppressed)
    assert "UP006" not in output
    assert "RUF100" not in output
