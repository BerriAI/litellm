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
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from fnmatch import fnmatch
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
# Written by scripts/mutation_diff_scope.py: the mutant-name globs this run was
# asked to execute. Absent for a whole-folder run, which has no such contract.
SCOPE_GLOBS_FILE = ROOT / "mutmut-scope-globs.txt"
# Overridable so the report can be regenerated outside CI, where the project venv
# `uv run --no-sync` expects may not exist.
MUTMUT_INVOCATION = shlex.split(
    os.environ.get("MUTMUT_CMD", "uv run --no-sync --with mutmut==3.5.0 mutmut")
)
# mutmut mangles a class method as `xǁ<Class>ǁ<method>` and a module-level
# function as `x_<function>`.
CLASS_NAME_SEPARATOR = "ǁ"
RESULT_LINE = re.compile(
    r"\s*(\S+):\s*(killed|survived|no tests|timeout|suspicious|skipped|not checked)\s*$"
)
MUTANT_NAME = re.compile(r"^(?P<module>.+)\.(?P<mangled>[^.]+)__mutmut_(?P<number>\d+)$")
COUNT_KEYS = ("killed", "survived", "no_tests", "skipped", "suspicious", "timeout", "segfault")


@dataclass(frozen=True)
class MutantName:
    """A parsed mutmut mutant identifier."""

    module: str
    class_name: str | None
    function: str
    number: str

    @property
    def mangled(self) -> str:
        """The name mutmut gives the mutated function inside the trampoline file."""
        if self.class_name is None:
            return f"x_{self.function}"
        return f"x{CLASS_NAME_SEPARATOR}{self.class_name}{CLASS_NAME_SEPARATOR}{self.function}"

    @property
    def qualified(self) -> str:
        return f"{self.class_name}.{self.function}" if self.class_name else self.function


def load_mutmut_config() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["mutmut"]


def get_results() -> tuple[tuple[str, str], ...]:
    """Parse `mutmut results` into (mutant name, status) pairs.

    A diff-scoped run leaves every out-of-scope mutant at `not checked`, so
    callers must drop that status before counting anything.
    """
    proc = subprocess.run(
        [*MUTMUT_INVOCATION, "results", "--all=true"], capture_output=True, text=True, check=False
    )
    matches = (RESULT_LINE.match(line) for line in proc.stdout.splitlines())
    return tuple((m.group(1), m.group(2)) for m in matches if m is not None)


def scope_globs() -> tuple[str, ...]:
    if not SCOPE_GLOBS_FILE.exists():
        return ()
    return tuple(
        stripped
        for line in SCOPE_GLOBS_FILE.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip())
    )


def unchecked_in_scope(
    results: tuple[tuple[str, str], ...], globs: tuple[str, ...]
) -> tuple[str, ...]:
    """In-scope mutants the run never got to.

    mutmut swallows an interrupt and still exits 0, so its exit code cannot tell a
    finished run from one that was killed halfway. What the run was asked to do is
    the only reliable contract: any requested mutant left at `not checked` means
    the score below covers less than the diff does.
    """
    return tuple(
        name
        for name, status in results
        if status == "not checked" and any(fnmatch(name, glob) for glob in globs)
    )


def summarize(results: tuple[tuple[str, str], ...]) -> dict | None:
    """Count only the mutants this run actually executed."""
    checked = tuple(status for _, status in results if status != "not checked")
    if not checked:
        return None
    return {
        "total": len(checked),
        **{
            key: sum(1 for status in checked if status == label)
            for key, label in (
                ("killed", "killed"),
                ("survived", "survived"),
                ("no_tests", "no tests"),
                ("skipped", "skipped"),
                ("suspicious", "suspicious"),
                ("timeout", "timeout"),
            )
        },
    }


def get_mutmut_show(mutant_name: str) -> str:
    proc = subprocess.run(
        [*MUTMUT_INVOCATION, "show", mutant_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or "(mutmut show produced no output)"


def parse_mutant_name(name: str) -> MutantName:
    """Parse a mutmut mutant identifier into its module, class, function and number.

    Module-level functions are `<dotted.module>.x_<function>__mutmut_<N>`; a
    function named `_foo` becomes `x__foo__mutmut_N`, so everything after the
    single `x_` prefix (leading underscores included) is the function name.
    Class methods are `<dotted.module>.xǁ<Class>ǁ<method>__mutmut_<N>`.

    An unrecognised name is returned verbatim as the function, so the report
    still shows something addressable instead of dropping the mutant.
    """
    m = MUTANT_NAME.match(name)
    if not m:
        return MutantName(module=name, class_name=None, function=name, number="?")
    mangled = m.group("mangled")
    if mangled.startswith(f"x{CLASS_NAME_SEPARATOR}"):
        parts = mangled.split(CLASS_NAME_SEPARATOR)
        if len(parts) == 3:
            return MutantName(
                module=m.group("module"),
                class_name=parts[1],
                function=parts[2],
                number=m.group("number"),
            )
    if mangled.startswith("x_"):
        return MutantName(
            module=m.group("module"),
            class_name=None,
            function=mangled[len("x_") :],
            number=m.group("number"),
        )
    return MutantName(module=name, class_name=None, function=name, number="?")


def function_anchor(module_path: str, function_name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", f"{module_path}-{function_name}".lower()).strip(
        "-"
    )


def module_to_file(module_path: str) -> Path | None:
    candidate = ROOT / Path(*module_path.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def find_function_in_file(
    file_path: Path, function_name: str, class_name: str | None = None
) -> tuple[int, int, str, list[int]] | None:
    """Find a function by name; returns the first match.

    Returns ``(start_line, end_line, source, all_match_lines)`` or ``None``.
    ``all_match_lines`` is the start line of every candidate definition. When
    ``len(all_match_lines) > 1`` the file defines the same name in several
    places and callers surface a disambiguation note. A mutant carrying class
    context is matched against that class's own methods, which is normally
    unambiguous.
    """
    src = file_path.read_text()
    tree = ast.parse(src)
    scopes = (
        [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if class_name is not None
        else [tree]
    )
    matches = [
        node
        for scope in scopes
        for node in ast.walk(scope)
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


def render_meta_style_mutant(mutant: MutantName) -> str | None:
    """Render the mutated function with `# MUTANT START`/`# MUTANT END` delimiters.

    Reads `mutants/<module>.py` (the trampoline file mutmut emits), finds
    `<mangled>__mutmut_orig` and `<mangled>__mutmut_<N>`, and renders the
    mutated version with the lines that differ from `__mutmut_orig` wrapped
    in `# MUTANT START`/`# MUTANT END` comments — the format from Meta's
    ACH paper (arXiv 2501.12862, Table 1).

    The function header is rewritten to use the original function name so
    the agent sees the source as it would appear in the file (rather than
    mutmut's internal `x_*__mutmut_<N>` name).

    Returns None if the trampoline file or either function cannot be found
    (the caller falls back to the unified diff).
    """
    trampoline = ROOT / "mutants" / Path(*mutant.module.split(".")).with_suffix(".py")
    if not trampoline.exists():
        return None

    src = trampoline.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    file_lines = src.splitlines()

    orig_def = f"{mutant.mangled}__mutmut_orig"
    mutant_def = f"{mutant.mangled}__mutmut_{mutant.number}"

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
    orig_lines[0] = orig_lines[0].replace(orig_def, mutant.function, 1)
    mutated_lines[0] = mutated_lines[0].replace(mutant_def, mutant.function, 1)

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


def render(
    config: dict,
    survivors: tuple[str, ...],
    stats: dict | None,
    unchecked: tuple[str, ...] = (),
) -> str:
    by_function: dict[tuple[str, str | None, str], list[tuple[str, MutantName]]] = defaultdict(list)
    for survivor in survivors:
        mutant = parse_mutant_name(survivor)
        by_function[(mutant.module, mutant.class_name, mutant.function)].append((survivor, mutant))

    out: list[str] = []
    out.append("# Mutation Test Report")
    out.append("")

    out.append("## Summary")
    out.append("")
    if stats:
        total = stats.get("total", 0) or sum(stats.get(k, 0) for k in COUNT_KEYS)
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
        out.append("- (no mutant results and no mutmut-cicd-stats.json)")
    if unchecked:
        out.append(f"- Never checked: **{len(unchecked)}** of the mutants this run asked for")
    out.append("")

    if unchecked:
        out.append(
            f"**The run stopped early: {len(unchecked)} in-scope mutants were never "
            "checked, so the score above covers less than the diff does. Treat this "
            "as a failed run.**"
        )
        out.append("")
    elif not stats:
        out.append(
            "**The mutation run produced no results at all. Treat this as a failed "
            "run, not as a passing one: nothing was executed to survive.**"
        )
        out.append("")

    if not survivors:
        if stats and not unchecked:
            out.append("**No surviving mutants — the test suite caught every mutation.**")
            out.append("")
        return "\n".join(out)

    out.append("## Surviving mutants by function")
    out.append("")
    for (module_path, class_name, function_name), items in by_function.items():
        qualified = items[0][1].qualified
        anchor = function_anchor(module_path, qualified)
        out.append(
            f"- [`{qualified}`](#{anchor}) — {len(items)} mutant"
            f"{'s' if len(items) != 1 else ''} ({module_path})"
        )
    out.append("")

    for (module_path, class_name, function_name), items in by_function.items():
        qualified = items[0][1].qualified
        anchor = function_anchor(module_path, qualified)
        out.append(f'<a id="{anchor}"></a>')
        out.append(f"## `{module_path}.{qualified}`")
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
            found = find_function_in_file(file_path, function_name, class_name)
            if found:
                start, end, fn_src, all_lines = found
                out.append(f"### Original function (lines {start}-{end})")
                out.append("")
                if len(all_lines) > 1:
                    line_list = ", ".join(str(line) for line in all_lines)
                    out.append(
                        f"> **Note:** {len(all_lines)} functions named "
                        f"`{function_name}` are defined in this file at lines "
                        f"{line_list}. Showing the first match; verify it is the "
                        f"one that was mutated before writing the killing test."
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
        for i, (mutant_name, mutant) in enumerate(items, 1):
            out.append(f"#### Mutation {i} of {len(items)} — `{mutant_name}`")
            out.append("")
            meta_style = render_meta_style_mutant(mutant)
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


def fallback_stats() -> dict | None:
    """mutmut's own export, used when `mutmut results` could not be parsed."""
    stats_file = ROOT / "mutants" / "mutmut-cicd-stats.json"
    if not stats_file.exists():
        return None
    try:
        exported = json.loads(stats_file.read_text())
    except json.JSONDecodeError as exc:
        print(f"warning: could not parse {stats_file}: {exc}", file=sys.stderr)
        return None
    return exported if any(exported.get(key, 0) for key in COUNT_KEYS) else None


def main() -> int:
    config = load_mutmut_config()

    results = get_results()
    stats = summarize(results) or fallback_stats()
    unchecked = unchecked_in_scope(results, scope_globs())
    survivors = tuple(name for name, status in results if status == "survived")
    report = render(config, survivors, stats, unchecked)

    out_path = ROOT / "mutation-report.md"
    out_path.write_text(report)
    print(
        f"Wrote {out_path} ({len(survivors)} survivor"
        f"{'s' if len(survivors) != 1 else ''}, {len(report)} chars)"
    )
    # Survivors are advisory. A run that checked nothing, or stopped partway through
    # what it was asked to check, is a broken run: reporting either as a clean sweep
    # is how a crashed mutmut turns into a green pull request.
    if stats is None:
        print(
            "error: mutmut reported no checked mutants; the run did not complete",
            file=sys.stderr,
        )
        return 1
    if unchecked:
        print(
            f"error: {len(unchecked)} in-scope mutants were never checked; "
            f"the run did not complete (first: {unchecked[0]})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
