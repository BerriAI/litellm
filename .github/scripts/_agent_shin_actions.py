"""Thin dry-run wrappers around every Agent Shin GitHub mutation.

Every destructive operation the rollout scripts perform (post a comment, close
a PR/issue, reopen, add/remove a label) is funneled through a `maybe_*` helper
here. The helpers take a single ``dry_run: bool`` keyword argument and the
body is intentionally trivial:

    if dry_run:
        print(...)        # log what we would do, return
        return
    real_mutation(...)    # otherwise, actually do it

That shape means a dry-run preview differs from the real run in exactly one
line per side effect: the call site. So when you `python3 script.py` locally
without ``--close``, you can be confident the actions printed are the ones the
GitHub Action would have performed (modulo ordering on retry/error paths,
which are deliberately simple).

Importing from this module pulls in the real mutations from
``triage_with_llm`` — call sites in the rollout scripts should NEVER import
``post_comment`` / ``close_pr`` / etc. directly; that would skip the dry-run
gate and is the bug class this module exists to prevent.
"""

from __future__ import annotations

import sys
import textwrap

# Import the module itself rather than the bare names so monkeypatching
# `triage_with_llm.post_comment` (or any of the other mutations) in tests is
# reflected here — `from triage_with_llm import post_comment` would bind the
# original function to a local name and bypass the patch, defeating the whole
# point of these wrappers.
import triage_with_llm


def _log(line: str) -> None:
    """Print a single dry-run line to stdout (one log statement per side effect)."""
    print(line, file=sys.stdout, flush=True)


def maybe_post_comment(repo: str, number: int, body: str, *, dry_run: bool) -> None:
    """Post a comment on ``repo#number`` — or, in dry-run, log what we would post."""
    if dry_run:
        _log(f"[DRY RUN] comment {repo}#{number}:")
        _log(textwrap.indent(body, "    "))
        return
    triage_with_llm.post_comment(repo, number, body)


def maybe_close_pr(repo: str, number: int, *, dry_run: bool) -> None:
    """Close ``repo#number`` (PR) — or log it."""
    if dry_run:
        _log(f"[DRY RUN] close PR {repo}#{number}")
        return
    triage_with_llm.close_pr(repo, number)


def maybe_close_issue(
    repo: str, number: int, *, dry_run: bool, not_planned: bool = True
) -> None:
    """Close ``repo#number`` (issue) — or log it. ``not_planned=True`` sets the
    standard ``state_reason`` for triage closures so they don't look like
    'completed'."""
    if dry_run:
        _log(
            f"[DRY RUN] close issue {repo}#{number}"
            f" (state_reason={'not_planned' if not_planned else 'completed'})"
        )
        return
    triage_with_llm.close_issue(repo, number, not_planned=not_planned)


def maybe_reopen_pr(repo: str, number: int, *, dry_run: bool) -> None:
    """Reopen a previously-closed PR — or log it."""
    if dry_run:
        _log(f"[DRY RUN] reopen PR {repo}#{number}")
        return
    triage_with_llm.reopen_pr(repo, number)


def maybe_reopen_issue(repo: str, number: int, *, dry_run: bool) -> None:
    """Reopen a previously-closed issue — or log it."""
    if dry_run:
        _log(f"[DRY RUN] reopen issue {repo}#{number}")
        return
    triage_with_llm.reopen_issue(repo, number)


def maybe_add_label(repo: str, number: int, label: str, *, dry_run: bool) -> None:
    """Add a label to ``repo#number`` — or log it."""
    if dry_run:
        _log(f"[DRY RUN] add label {label!r} to {repo}#{number}")
        return
    triage_with_llm.add_label(repo, number, label)


def maybe_remove_label(repo: str, number: int, label: str, *, dry_run: bool) -> None:
    """Remove a label from ``repo#number`` — or log it. A missing label is not
    an error in the real path either."""
    if dry_run:
        _log(f"[DRY RUN] remove label {label!r} from {repo}#{number}")
        return
    triage_with_llm.remove_label(repo, number, label)
