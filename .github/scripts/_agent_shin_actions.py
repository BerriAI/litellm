"""Dry-run wrapper(s) around Agent Shin GitHub mutations.

The rollout scripts currently need only one mutation wrapped, so this module
exposes a single ``maybe_post_comment`` helper. It takes a ``dry_run: bool``
keyword argument and the body is intentionally trivial:

    if dry_run:
        print(...)        # log what we would do, return
        return
    real_mutation(...)    # otherwise, actually do it

That shape means a dry-run preview differs from the real run in exactly one
line per side effect: the call site. So when you `python3 script.py` locally
without ``--close``, you can be confident the actions printed are the ones the
GitHub Action would have performed (modulo ordering on retry/error paths,
which are deliberately simple). Any further mutation a rollout script needs
should get the same ``maybe_*`` treatment instead of calling the raw
``triage_with_llm`` mutation directly.

Importing from this module pulls in the real mutation from ``triage_with_llm``
— call sites in the rollout scripts should NEVER import ``post_comment``
directly; that would skip the dry-run gate and is the bug class this module
exists to prevent.
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
