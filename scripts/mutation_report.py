#!/usr/bin/env python3
"""Generate an agent-actionable mutation testing report.

Reads the mutmut sandbox state at `mutants/` and produces a single
`mutation-report.md` grouped by function. For each function with surviving
mutants, the report embeds the original function source (via AST), the
unified diff for each surviving mutation (via `mutmut show`), and the
existing test file(s) — followed by an ACH-style instruction asking the
reader to write tests that kill the survivors.

Run after `mutmut run` and `mutmut export-cicd-stats`. Expects mutmut to be
invokable as `uv run --no-sync --with mutmut==<version> mutmut <subcommand>`.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
MUTMUT_INVOCATION = ["uv", "run", "--no-sync", "--with", "mutmut==3.5.0", "mutmut"]


def load_mutmut_config() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["mutmut"]


def get_survivors() -> list[str]:
    proc = subprocess.run(
        [*MUTMUT_INVOCATION, "results"], capture_output=True, text=True, check=False
    )
    survivors = []
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*(\S+):\s*survived\s*$", line)
        if m:
            survivors.append(m.group(1))
    return survivors


def get_mutmut_show(mutant_name: str) -> str:
    proc = subprocess.run(
        [*MUTMUT_INVOCATION, "show", mutant_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or "(mutmut show produced no output)"


def parse_mutant_name(name: str) -> tuple[str, str, str]:
    """Parse `<dotted.module>.x_<function>__mutmut_<N>` -> (module, function, N).

    mutmut prefixes mutated functions with `x_` (single underscore). For a
    function named `foo`, mutants are `x_foo__mutmut_N`. For a function named
    `_foo` (leading underscore), the mutant becomes `x__foo__mutmut_N` — so
    the regex matches a single underscore after `x` and captures everything
    (including any leading underscores) up to `__mutmut_<N>`.
    """
    m = re.match(r"^(.+)\.x_(.+)__mutmut_(\d+)$", name)
    if not m:
        return name, name, "?"
    return m.group(1), m.group(2), m.group(3)


def module_to_file(module_path: str) -> Path | None:
    candidate = ROOT / Path(*module_path.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def find_function_in_file(
    file_path: Path, function_name: str
) -> tuple[int, int, str] | None:
    src = file_path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            lines = src.splitlines()
            return (
                node.lineno,
                node.end_lineno,
                "\n".join(lines[node.lineno - 1 : node.end_lineno]),
            )
    return None


def collect_test_files(tests_dir: list[str]) -> list[Path]:
    found: list[Path] = []
    for entry in tests_dir:
        p = ROOT / entry
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("test_*.py")))
    return found


def render(config: dict, survivors: list[str], stats: dict | None) -> str:
    by_function: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for survivor in survivors:
        module_path, function_name, mutant_num = parse_mutant_name(survivor)
        by_function[(module_path, function_name)].append((survivor, mutant_num))

    out: list[str] = []
    out.append("# Mutation Test Report")
    out.append("")

    out.append("## Summary")
    out.append("")
    if stats:
        total = stats.get("total", 0) or sum(
            stats.get(k, 0)
            for k in (
                "killed",
                "survived",
                "no_tests",
                "skipped",
                "suspicious",
                "timeout",
                "segfault",
            )
        )
        killed = stats.get("killed", 0)
        survived = stats.get("survived", 0)
        score = (killed / total * 100) if total else 0.0
        out.append(f"- Total mutants: **{total}**")
        out.append(f"- Killed: **{killed}**")
        out.append(f"- Survived: **{survived}**")
        out.append(f"- Mutation score: **{score:.1f}%**")
        for k in ("no_tests", "skipped", "suspicious", "timeout", "segfault"):
            v = stats.get(k, 0)
            if v:
                out.append(f"- {k.replace('_', ' ').title()}: {v}")
    else:
        out.append(f"- Survivors found: **{len(survivors)}**")
        out.append("- (mutmut-cicd-stats.json not available — full counts unavailable)")
    out.append("")

    if not survivors:
        out.append("**No surviving mutants — the test suite caught every mutation.**")
        out.append("")
        return "\n".join(out)

    out.append("## Surviving mutants by function")
    out.append("")
    for (module_path, function_name), items in by_function.items():
        anchor = f"{function_name.lower().replace('_', '-')}"
        out.append(
            f"- [`{function_name}`](#{anchor}) — {len(items)} mutant"
            f"{'s' if len(items) != 1 else ''} ({module_path})"
        )
    out.append("")

    for (module_path, function_name), items in by_function.items():
        out.append(f"## `{function_name}`")
        out.append("")
        out.append(f"**Module:** `{module_path}`")

        file_path = module_to_file(module_path)
        if file_path is None:
            out.append("")
            out.append(f"_(could not locate source file for module `{module_path}`)_")
            out.append("")
        else:
            rel = file_path.relative_to(ROOT)
            out.append(f"**File:** `{rel}`")
            out.append("")
            found = find_function_in_file(file_path, function_name)
            if found:
                start, end, fn_src = found
                out.append(f"### Original function (lines {start}-{end})")
                out.append("")
                out.append("```python")
                out.append(fn_src)
                out.append("```")
                out.append("")
            else:
                out.append(f"_(could not locate `{function_name}` in {rel} via AST)_")
                out.append("")

        out.append(f"### Surviving mutations ({len(items)})")
        out.append("")
        for i, (mutant_name, mutant_num) in enumerate(items, 1):
            out.append(f"#### Mutation {i} of {len(items)} — `{mutant_name}`")
            out.append("")
            out.append("```diff")
            out.append(get_mutmut_show(mutant_name))
            out.append("```")
            out.append("")

    test_files = collect_test_files(config.get("tests_dir", []))
    if test_files:
        out.append("## Existing tests")
        out.append("")
        out.append(
            "These are the test files that mutmut considered when classifying the "
            "mutants above. New tests should be added here, matching existing "
            "conventions, fixtures, and naming."
        )
        out.append("")
        for tf in test_files:
            rel = tf.relative_to(ROOT)
            out.append(f"### `{rel}`")
            out.append("")
            out.append("```python")
            out.append(tf.read_text())
            out.append("```")
            out.append("")

    out.append("## Task")
    out.append("")
    out.append(
        dedent(
            """\
            For each surviving mutant listed above, write a new test in the
            existing test file (matching its conventions, fixtures, and naming
            style) that:

            - **Fails** when the mutated version of the function is in place.
            - **Passes** when the original (correct) version is in place.

            Aim for one test per surviving mutant. If multiple mutants in the
            same function can be killed by a single test, that is fine — note
            which mutant numbers in the test name or docstring.

            Do not modify the source file. Only add tests.
            """
        ).strip()
    )
    out.append("")

    return "\n".join(out)


def main() -> int:
    config = load_mutmut_config()

    stats_file = ROOT / "mutants" / "mutmut-cicd-stats.json"
    stats: dict | None = None
    if stats_file.exists():
        try:
            stats = json.loads(stats_file.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: could not parse {stats_file}: {exc}", file=sys.stderr)

    survivors = get_survivors()
    report = render(config, survivors, stats)

    out_path = ROOT / "mutation-report.md"
    out_path.write_text(report)
    print(
        f"Wrote {out_path} ({len(survivors)} survivor"
        f"{'s' if len(survivors) != 1 else ''}, {len(report)} chars)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
