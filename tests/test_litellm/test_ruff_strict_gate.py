import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import lint_base_counts
import ruff_strict_gate as gate

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]

Violation = gate.Violation

_ENABLED_BY_RUFF_DEFAULTS = frozenset({"F401"})


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


@_needs_ruff
def test_the_checker_identity_is_rekeyed_by_either_ruff_config_or_the_ruff_version():
    identity = gate.checker_identity()
    assert identity.name == "ruff-strict"
    assert identity.fingerprints == (
        lint_base_counts.sha256_of(gate.STRICT_CONFIG),
        lint_base_counts.sha256_of(gate.BASE_CONFIG),
        gate.ruff_version(),
    )


def test_every_headroom_rule_is_one_the_strict_config_selects():
    selectors = tuple(_strict_selected())
    unmeasured = frozenset(code for code in gate.HEADROOM if not code.startswith(selectors))
    assert unmeasured == frozenset(), (
        f"the gate never counts {sorted(unmeasured)}, so headroom for them is dead config that reads "
        "as coverage. Either select them in ruff-strict.toml or drop the headroom entry."
    )
