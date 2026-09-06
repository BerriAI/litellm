#!/usr/bin/env python3
"""Claude Code PreToolUse hook gating PR creation on a passing /live-pr-risk run.

A passing run of the live-pr-risk skill (see .claude/skills/live-pr-risk/) writes
"PASS <head-sha>" to `git rev-parse --git-path live-pr-risk-pass`. This hook denies
any PR-creating tool call unless that marker matches the repo's current HEAD, so a
new commit stales the marker and requires a fresh run.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

GH_PR_CREATE: Final = re.compile(r"\bgh\s+pr\s+create\b")
GH_API_PULLS: Final = re.compile(r"\bgh\s+api\b[^|;&]*\bpulls\b")
GH_API_WRITE: Final = re.compile(r"(\s-X\s*POST\b|\s--method[=\s]*POST\b|\s-[fF]\s|\s--(raw-)?field\b)")
MCP_PR_CREATE: Final = re.compile(r"create_pull_request")
MARKER_NAME: Final = "live-pr-risk-pass"


def is_pr_creation(tool_name: str, command: str) -> bool:
    if tool_name.startswith("mcp__"):
        return bool(MCP_PR_CREATE.search(tool_name))
    if tool_name != "Bash":
        return False
    if GH_PR_CREATE.search(command):
        return True
    return bool(GH_API_PULLS.search(command) and GH_API_WRITE.search(command))


def git_output(args: tuple[str, ...], cwd: Path) -> str | None:
    result: Final = subprocess.run(("git", *args), cwd=cwd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def marker_matches_head(project_dir: Path) -> bool:
    head_sha: Final = git_output(("rev-parse", "HEAD"), project_dir)
    marker_path: Final = git_output(("rev-parse", "--path-format=absolute", "--git-path", MARKER_NAME), project_dir)
    if head_sha is None or marker_path is None:
        return False
    try:
        marker: Final = Path(marker_path).read_text().strip()
    except OSError:
        return False
    return marker == f"PASS {head_sha}"


def deny(reason: str) -> None:
    print(  # noqa: T201  # hook protocol: the JSON verdict must go to stdout
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main() -> None:
    try:
        payload: Final = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    tool_name: Final = str(payload.get("tool_name", ""))
    tool_input: Final = payload.get("tool_input") or {}
    command: Final = str(tool_input.get("command", "")) if isinstance(tool_input, dict) else ""
    if not is_pr_creation(tool_name, command):
        return
    project_dir: Final = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    if marker_matches_head(project_dir):
        return
    deny(
        "PR creation is gated on a passing /live-pr-risk run for the exact HEAD you are "
        "shipping. Run the live-pr-risk skill (.claude/skills/live-pr-risk/) on this "
        "branch; a pass (zero Breaking findings) records 'PASS <head-sha>' in "
        "`git rev-parse --git-path live-pr-risk-pass`. The marker is missing or stale, "
        "which also happens after any new commit. Never write the marker without a "
        "passing run; fix the findings and re-run instead."
    )


if __name__ == "__main__":
    main()
