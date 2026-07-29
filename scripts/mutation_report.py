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
from difflib import SequenceMatcher
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


def function_anchor(module_path: str, function_name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", f"{module_path}-{function_name}".lower()).strip(
        "-"
    )


def module_to_file(module_path: str) -> Path | None:
    candidate = ROOT / Path(*module_path.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def find_function_in_file(
    file_path: Path, function_name: str
) -> tuple[int, int, str, list[int]] | None:
    """Find a top-level or nested function by name; returns the first match.

    Returns ``(start_line, end_line, source, all_match_lines)`` or ``None``.
    ``all_match_lines`` is the start line of every function (any nesting
    level) in the file with this name. When ``len(all_match_lines) > 1`` the
    file defines the same name in multiple places (e.g., a module-level
    helper and a class method) — mutmut's mutant identifier does not carry
    class context, so we can't determine which definition was mutated.
    Callers surface a disambiguation note in that case.
    """
    src = file_path.read_text()
    tree = ast.parse(src)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if not matches:
        return None
    first = matches[0]
    lines = src.splitlines()
    return (
        first.lineno,
        first.end_lineno,
        "\n".join(lines[first.lineno - 1 : first.end_lineno]),
        [m.lineno for m in matches],
    )


def collect_test_files(tests_dir: list[str]) -> list[Path]:
    found: list[Path] = []
    for entry in tests_dir:
        p = ROOT / entry
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("test_*.py")))
    return found


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def render_meta_style_mutant(
    module_path: str, function_name: str, mutant_num: str
) -> str | None:
    """Render the mutated function with `# MUTANT START`/`# MUTANT END` delimiters.

    Reads `mutants/<module>.py` (the trampoline file mutmut emits), finds
    `x_<func>__mutmut_orig` and `x_<func>__mutmut_<N>`, and renders the
    mutated version with the lines that differ from `__mutmut_orig` wrapped
    in `# MUTANT START`/`# MUTANT END` comments — the format from Meta's
    ACH paper (arXiv 2501.12862, Table 1).

    The function header is rewritten to use the original function name so
    the agent sees the source as it would appear in the file (rather than
    mutmut's internal `x_*__mutmut_<N>` name).

    Returns None if the trampoline file or either function cannot be found
    (the caller falls back to the unified diff).
    """
    trampoline = ROOT / "mutants" / Path(*module_path.split(".")).with_suffix(".py")
    if not trampoline.exists():
        return None

    src = trampoline.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    file_lines = src.splitlines()

    orig_def = f"x_{function_name}__mutmut_orig"
    mutant_def = f"x_{function_name}__mutmut_{mutant_num}"

    orig_node = mutated_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == orig_def:
                orig_node = node
            elif node.name == mutant_def:
                mutated_node = node

    if orig_node is None or mutated_node is None:
        return None

    orig_lines = file_lines[orig_node.lineno - 1 : orig_node.end_lineno]
    mutated_lines = file_lines[mutated_node.lineno - 1 : mutated_node.end_lineno]
    if not orig_lines or not mutated_lines:
        return None

    # Rewrite the def line to use the original (non-trampolined) function name
    # so the agent sees the function as it appears in the source file.
    orig_lines[0] = orig_lines[0].replace(orig_def, function_name, 1)
    mutated_lines[0] = mutated_lines[0].replace(mutant_def, function_name, 1)

    matcher = SequenceMatcher(a=orig_lines, b=mutated_lines)
    out: list[str] = []
    in_diff = False

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            if in_diff:
                # Close the block at the indent of the line just inside it.
                indent = _indent_of(out[-1]) if out else ""
                out.append(f"{indent}# MUTANT END")
                in_diff = False
            out.extend(mutated_lines[j1:j2])
        else:
            if not in_diff:
                # Open the block at the indent of the first differing line.
                if j1 < len(mutated_lines):
                    indent = _indent_of(mutated_lines[j1])
                elif i1 < len(orig_lines):
                    indent = _indent_of(orig_lines[i1])
                else:
                    indent = ""
                out.append(f"{indent}# MUTANT START")
                in_diff = True
            if op == "delete":
                # Mutation removed lines — surface what was deleted as a
                # comment so the agent can see the intent of the change.
                for deleted in orig_lines[i1:i2]:
                    indent = _indent_of(deleted)
                    out.append(f"{indent}# (deleted by mutation): {deleted.lstrip()}")
            else:
                # replace / insert: take from mutated_lines
                out.extend(mutated_lines[j1:j2])

    if in_diff:
        indent = _indent_of(out[-1]) if out else ""
        out.append(f"{indent}# MUTANT END")

    return "\n".join(out)


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
        anchor = function_anchor(module_path, function_name)
        out.append(
            f"- [`{function_name}`](#{anchor}) — {len(items)} mutant"
            f"{'s' if len(items) != 1 else ''} ({module_path})"
        )
    out.append("")

    for (module_path, function_name), items in by_function.items():
        anchor = function_anchor(module_path, function_name)
        out.append(f'<a id="{anchor}"></a>')
        out.append(f"## `{module_path}.{function_name}`")
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
                start, end, fn_src, all_lines = found
                out.append(f"### Original function (lines {start}-{end})")
                out.append("")
                if len(all_lines) > 1:
                    line_list = ", ".join(str(line) for line in all_lines)
                    out.append(
                        f"> **Note:** {len(all_lines)} functions named "
                        f"`{function_name}` are defined in this file at lines "
                        f"{line_list}. Showing the first match. mutmut's "
                        f"mutant identifier does not carry class context, so "
                        f"the body below may not correspond to the function "
                        f"that was actually mutated — verify manually before "
                        f"writing the killing test."
                    )
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
            meta_style = render_meta_style_mutant(
                module_path, function_name, mutant_num
            )
            if meta_style is not None:
                out.append(
                    "Mutated function (the bug is delimited by "
                    "`# MUTANT START` / `# MUTANT END`):"
                )
                out.append("")
                out.append("```python")
                out.append(meta_style)
                out.append("```")
                out.append("")
                out.append("<details><summary>Unified diff (`mutmut show`)</summary>")
                out.append("")
                out.append("```diff")
                out.append(get_mutmut_show(mutant_name))
                out.append("```")
                out.append("")
                out.append("</details>")
                out.append("")
            else:
                # Fallback: trampoline file or function lookup failed.
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
