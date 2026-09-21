#!/usr/bin/env python3
"""Three invariants about what lives in .github/workflows/ and what its names mean.

`.github/workflows/` is a directory GitHub reads, not a place to keep things. Every
file at its top level is parsed as a workflow, so a script or a data file parked there
is either an invalid workflow or an orphan nobody can find. A subdirectory is not read
at all, so helper files may live in one. GitHub accepts both `.yml` and `.yaml`, and
this repo spells them `.yml`, which is a naming rule rather than a validity one and is
reported separately. And the `_` prefix is the repo's only signal that a workflow is a
reusable building block rather than something that runs on its own, which is worth
nothing unless it is true both ways.

    WF001   a top-level file in .github/workflows/ that is not a workflow at all
    WF002   a workflow whose only trigger is `workflow_call` but is not `_`-prefixed
    WF003   a `_`-prefixed workflow that no other workflow can call
    WF004   a real workflow spelled `.yaml` where this directory spells them `.yml`

A workflow with `workflow_call` alongside a human trigger is deliberately dual-mode
and belongs under its plain name, so only the call-only ones are held to WF002.

Usage
-----
    python assert_workflow_dir_hygiene.py

Exit code 1 if any violation is found.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Final

import yaml

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_DIR: Final = REPO_ROOT / ".github" / "workflows"
SCRIPT_HOME: Final = ".github/scripts/"
REUSABLE_PREFIX: Final = "_"
CALL_TRIGGER: Final = "workflow_call"
CANONICAL_SUFFIX: Final = ".yml"
WORKFLOW_SUFFIXES: Final = frozenset((CANONICAL_SUFFIX, ".yaml"))


@dataclass(frozen=True, slots=True)
class Finding:
    subject: str
    code: str
    detail: str

    def render(self) -> str:
        return f"  - {self.subject}: {self.code} {self.detail}"


def _triggers(document: object) -> frozenset[str]:
    if not isinstance(document, dict):
        return frozenset()
    raw: Final = document.get("on", document.get(True))
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, dict):
        return frozenset(str(key) for key in raw)
    if isinstance(raw, list):
        return frozenset(str(item) for item in raw)
    return frozenset()


def _workflows(directory: pathlib.Path) -> tuple[pathlib.Path, ...]:
    return tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in WORKFLOW_SUFFIXES
    )


def _strays(directory: pathlib.Path) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            path.name,
            "WF001",
            f"is not a workflow, and GitHub parses every top-level file here as one; "
            f"move it to {SCRIPT_HOME} or into a subdirectory, which GitHub does not read",
        )
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix not in WORKFLOW_SUFFIXES
    )


def _misspelled(directory: pathlib.Path) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            path.name,
            "WF004",
            f"is a real workflow and GitHub reads it, but this directory spells them "
            f"{CANONICAL_SUFFIX}; rename it to {path.stem}{CANONICAL_SUFFIX}",
        )
        for path in _workflows(directory)
        if path.suffix != CANONICAL_SUFFIX
    )


def _misnamed(directory: pathlib.Path) -> tuple[Finding, ...]:
    return tuple(
        finding
        for path in _workflows(directory)
        for finding in _naming_findings(path, _triggers(yaml.safe_load(path.read_text(encoding="utf-8"))))
    )


def _naming_findings(path: pathlib.Path, triggers: frozenset[str]) -> tuple[Finding, ...]:
    underscored: Final = path.name.startswith(REUSABLE_PREFIX)
    if triggers == frozenset({CALL_TRIGGER}) and not underscored:
        return (
            Finding(
                path.name,
                "WF002",
                f"is only callable by another workflow, so name it {REUSABLE_PREFIX}{path.name}",
            ),
        )
    if underscored and CALL_TRIGGER not in triggers:
        return (
            Finding(
                path.name,
                "WF003",
                f"is named as a reusable workflow but has no {CALL_TRIGGER} trigger; "
                "add one or drop the prefix",
            ),
        )
    return ()


def main() -> int:
    findings: Final = _strays(WORKFLOW_DIR) + _misspelled(WORKFLOW_DIR) + _misnamed(WORKFLOW_DIR)
    if not findings:
        total: Final = len(_workflows(WORKFLOW_DIR))
        sys.stdout.write(
            f"OK: {total} workflows, every file in .github/workflows/ is one, and the "
            f"{REUSABLE_PREFIX} prefix means callable in both directions.\n"
        )
        return 0
    sys.stdout.write("ERROR: .github/workflows/ holds files that break its own conventions\n")
    for finding in findings:
        sys.stdout.write(f"{finding.render()}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
