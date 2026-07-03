"""Regression tests for CircleCI change-based job gating.

`.circleci/scripts/classify_changes.sh` is the pure decision function behind
`path_filter.sh`: given the list of files a PR changed (on stdin) and a job
category, it prints `run` or `skip`. The gating contract we lock in here:

  * docs-only changes (``*.md``, ``*.mdx``, ``docs/``) run nothing
  * client-only changes (``ui/``) run client jobs but skip backend jobs
  * any backend change runs both client and backend jobs

If this logic silently regresses, real test jobs get skipped, so these cases
are the guardrail against that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".circleci" / "scripts" / "classify_changes.sh"


def classify(category: str, changed: list[str]) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), category],
        input="\n".join(changed),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


DOCS = ["README.md", "docs/my_website/index.mdx", "litellm/anywhere.md"]
CLIENT = ["ui/litellm-dashboard/src/App.tsx"]
BACKEND = ["litellm/main.py"]


@pytest.mark.parametrize(
    "category,changed,expected",
    [
        # docs-only: skip everything
        ("backend", DOCS, "skip"),
        ("client", DOCS, "skip"),
        ("backend", [], "skip"),
        ("client", [], "skip"),
        # client-only: backend skips, client runs
        ("backend", CLIENT, "skip"),
        ("client", CLIENT, "run"),
        ("backend", CLIENT + DOCS, "skip"),
        ("client", CLIENT + DOCS, "run"),
        # any backend change: both run ("backend runs both")
        ("backend", BACKEND, "run"),
        ("client", BACKEND, "run"),
        ("backend", BACKEND + DOCS, "run"),
        ("client", BACKEND + DOCS, "run"),
        ("backend", BACKEND + CLIENT, "run"),
        ("client", BACKEND + CLIENT, "run"),
    ],
)
def test_classify_decisions(category: str, changed: list[str], expected: str) -> None:
    assert classify(category, changed) == expected


def test_markdown_under_ui_counts_as_client_not_docs() -> None:
    assert classify("client", ["ui/litellm-dashboard/README.md"]) == "run"
    assert classify("backend", ["ui/litellm-dashboard/README.md"]) == "skip"


def test_non_docs_directory_with_docs_in_name_is_backend() -> None:
    assert classify("backend", ["documentation_tests/foo.py"]) == "run"


def test_unknown_category_fails_open_to_run() -> None:
    assert classify("mystery", DOCS) == "run"
