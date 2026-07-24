#!/usr/bin/env python3
"""One-shot 7-day heads-up sweep for the Agent Shin rollout.

Posts a friendly "the OSS triage bot kicks in next Monday" comment on every
open external PR/issue that currently *would* fail the new rubric — i.e.,
every PR/issue Agent Shin would close once the rollout completes. The point
is to give contributors a full week to fix their description before the bot
ever takes a destructive action, so nobody is surprised by an auto-close.

The script is designed to run **exactly once** at rollout, fired by a manual
``workflow_dispatch`` (``dry_run=false``) on the heads-up workflow. Re-runs
are safe: every comment is stamped with the hidden ``HEADS_UP_MARKER`` and
PRs/issues that already carry the marker are skipped.

Dry-run vs. real run
--------------------
Defaults to dry-run. Passing ``--close`` flips into real mode. Every GitHub
mutation goes through ``_agent_shin_actions``, which has a one-line
``if dry_run: log else: do_it`` per call, so the only difference between a
dry-run preview and the real run is the call site that actually hits the
GitHub API.

Local preview::

    python3 .github/scripts/triage_rollout_heads_up.py --repo BerriAI/litellm

Real run (the manual rollout dispatch uses this)::

    python3 .github/scripts/triage_rollout_heads_up.py --repo BerriAI/litellm --close
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the sibling triage_with_llm + _agent_shin_actions importable when this
# script is invoked directly (the GitHub workflow does `python3 .github/scripts/...`).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _agent_shin_actions import maybe_post_comment  # noqa: E402
from agent_shin_shared import (  # noqa: E402
    AGENT_SHIN_DEFAULT_BOT_LOGIN,
    ALLOWLIST_LOGINS,
    list_open_items,
)
from triage_with_llm import (  # noqa: E402
    DEFAULT_MODEL,
    call_llm_judge,
    fetch_issue,
    fetch_pr,
    gh,
    is_internal_contributor,
    review_gate,
    triage,
)

# Hidden marker so re-runs skip PRs/issues we've already notified. Distinct from
# the within-grace / ready / regressed markers so it can't be confused with the
# steady-state lifecycle comments.
HEADS_UP_MARKER = "<!-- agent-shin:rollout-heads-up -->"

# Placeholder until the litellm-docs PR ships. The rollout blog post explains
# the new rubric, the 7-day grace, and how to recover after an auto-close.
# TODO(docs): replace with the canonical URL once the litellm-docs PR merges.
ROLLOUT_BLOG_URL = "https://docs.litellm.ai/docs/agent_shin_triage_rollout"

# Default cutoff is one week from "now". Computed at runtime so the wording
# stays correct even if the rollout is merged later than planned. The user can
# override with --close-on YYYY-MM-DD when running the script manually.
DEFAULT_GRACE_DAYS = 7

# The daily auto-close sweeps (close_low_quality_prs.yml at 09:00 UTC and
# review_gate.yml at 09:30 UTC) are what actually close a still-failing item,
# so the deadline we promise contributors has to name that wall-clock moment.
ACTIVATION_TIME_UTC = "09:00 UTC"


def _format_cutoff(cutoff: dt.date) -> str:
    """Human-readable, timezone-explicit cutoff, e.g. ``Monday, June 1, 2026
    (09:00 UTC)`` — the moment a still-failing PR/issue gets closed."""
    return (
        f"{cutoff.strftime('%A, %B')} {cutoff.day}, {cutoff.year} "
        f"({ACTIVATION_TIME_UTC})"
    )


def _rubric_section_pr() -> str:
    return (
        "**Going forward, every external PR needs ONE of:**\n"
        "\n"
        "- A linked GitHub issue using a closing keyword: "
        "`Fixes #1234`, `Closes #1234`, or `Resolves #1234`, OR\n"
        "- All three of: a clear **problem description**, **expected vs. "
        "actual behavior**, and **end-to-end QA proof** (at least one of a "
        "short screen recording / video, before/after screenshots, or the "
        "exact commands you ran with their real output; mocked or stubbed "
        "runs don't count).\n"
        "\n"
        "PRs also need a **Greptile confidence score of 4/5 or higher** before "
        "the bot will tag them `ready for review`. You can `@greptileai` to "
        "request a fresh review at any time, including after the PR is closed."
    )


def _rubric_section_issue() -> str:
    return (
        "**Going forward, every external issue needs:**\n"
        "\n"
        "- For **bug reports**: end-to-end evidence of the bug (at least one "
        "of a screen recording / video, a screenshot, or the exact commands "
        "you ran with their real output / traceback) plus expected vs. actual "
        "behavior. Written steps with no run output don't count, and mocked "
        "or stubbed runs don't count.\n"
        "- For **feature requests**: a clear description of the proposed "
        "feature plus a use case + concrete example (config, API call, UI "
        "flow, or scenario showing what's blocked today)."
    )


def _description_only_note(kind: str) -> str:
    noun = "PR" if kind == "pr" else "issue"
    return (
        f"⚠️ **The requirements must live in the {noun} *description*, not in "
        "comments.** Some PRs/issues collect 100+ comments from humans and "
        "bots; reading the entire thread on every triage run would balloon "
        "GitHub API usage (we'd start getting 429'd) and blow out the LLM "
        "judge's context. The bot only reads the description, so anything "
        "you add as a comment will be invisible to it."
    )


def _missing_section(verdict: dict, greptile_score: int | None) -> str:
    """Bullet list of what's currently missing on this PR/issue.

    Combines the LLM judge's `missing` list (rubric items) with a Greptile
    shortfall (for PRs) so the contributor sees one list of things to fix.
    """
    missing = list(verdict.get("missing") or [])
    if greptile_score is not None and greptile_score < 4:
        missing.insert(
            0,
            f"Greptile's most recent review scored this PR {greptile_score}/5 "
            "(below the 4/5 bar Agent Shin will require).",
        )
    if not missing:
        return (
            "_The bot couldn't articulate a specific missing piece; see the "
            "rubric link above and double-check the description includes all "
            "of it before the rollout._"
        )
    bullets = "\n".join(f"- {m}" for m in missing)
    return f"**What this one is currently missing:**\n\n{bullets}"


def _recovery_section(kind: str) -> str:
    if kind == "pr":
        return (
            "**If the bot closes this PR after the rollout:** update the "
            "description with the missing pieces, then either open a fresh "
            "PR or comment `@agent-shin reconsider` on the closed PR. If "
            "Greptile re-scores you at 4/5 or higher I'll reopen and tag "
            "the PR `ready for review`. (`@greptileai` works on closed PRs "
            "too; a fresh review is one of the signals that lifts you back "
            "into the queue.) This is **not** us losing interest in your "
            "change; far from it. We just need open PRs to be a list of "
            "things a maintainer can act on, so we can get to yours faster."
        )
    return (
        "**If the bot closes this issue after the rollout:** edit the issue "
        "description to add the missing pieces, then comment `@agent-shin "
        "reconsider` on the closed issue. I'll re-evaluate and, if the rubric "
        "is met, reopen it. (GitHub doesn't let external authors reopen an "
        "issue a maintainer or bot closed, so the comment is the reliable "
        "path.) This is **not** us saying the bug isn't real or the request "
        "isn't useful; it's so the remaining open issues are a list of things "
        "a maintainer can act on."
    )


def format_heads_up_comment(
    *, kind: str, verdict: dict, greptile_score: int | None, cutoff: dt.date
) -> str:
    """Compose the friendly 7-day heads-up comment posted on a failing PR/issue."""
    noun = "PR" if kind == "pr" else "issue"
    rubric = _rubric_section_pr() if kind == "pr" else _rubric_section_issue()
    cutoff_str = _format_cutoff(cutoff)
    explanation = (verdict.get("explanation") or "").strip()
    explanation_block = (
        f"> _(The judge's note for this one: {explanation})_\n\n" if explanation else ""
    )

    return (
        "🚅 **Heads-up: we're turning on the OSS triage bot in "
        f"{DEFAULT_GRACE_DAYS} days, on {cutoff_str}.**\n"
        "\n"
        "We're rolling out **Agent Shin**, an LLM-as-judge triage bot for "
        f"external {noun}s. Once it's live, the bot reads each open "
        f"{noun}'s description, scores it against a small rubric, and "
        f"auto-closes any {noun} that's missing the basics, with a single "
        f"comment explaining what's missing and how to recover. Full "
        f"context: [Agent Shin rollout blog post]({ROLLOUT_BLOG_URL}).\n"
        "\n"
        f"{rubric}\n"
        "\n"
        f"{_description_only_note(kind)}\n"
        "\n"
        f"{_missing_section(verdict, greptile_score)}\n"
        "\n"
        f"{explanation_block}"
        "**Timeline (you have a week):**\n"
        "\n"
        f"- We turn the bot on in {DEFAULT_GRACE_DAYS} days, on "
        f"**{cutoff_str}**. You have until then to update this {noun}'s "
        "description with the missing pieces above.\n"
        f"- If this {noun} still fails the rubric at **{cutoff_str}**, "
        "we'll close it.\n"
        f"- From then on the bot runs daily, and every {noun} that fails "
        "the rubric gets a **2-hour lifetime**: one warning comment, then "
        "auto-close 2 hours later.\n"
        "\n"
        f"{_recovery_section(kind)}\n"
        "\n"
        f"{HEADS_UP_MARKER}"
    )


def _list_open_numbers(repo: str, kind: str) -> list[int]:
    """Return every open PR or issue number in ``repo``.

    Delegates to ``list_open_items`` so the full backlog is fetched (no cap)
    and the `gh {pr,issue} list` invocation stays in one shared place. ``gh
    issue list`` would include PRs, but ``list_open_items`` uses the dedicated
    command per kind, so the two never mix.
    """
    return [
        item["number"] for item in list_open_items(kind, repo=repo, fields="number")
    ]


def _has_heads_up_marker(item: dict) -> bool:
    """Cheap fast-path: check the PR/issue body itself for the marker.

    The marker is appended to the *comment* we post, not the body, so this
    will only fire if the body literally contains the marker text. We still
    do the comment-marker check separately below; this body check just lets
    us short-circuit for PRs/issues that quote the marker for any reason.
    """
    body = item.get("body") or ""
    return HEADS_UP_MARKER in body


def _comments_have_marker(repo: str, number: int) -> bool:
    """True if the bot already posted a comment carrying the marker.

    Used for idempotency: a re-run skips items the previous run notified.
    Filters by author (matching the sibling marker-checks in
    ``triage_with_llm._has_marker`` and
    ``agent_shin_shared.seconds_since_latest_marker_comment``) so a
    contributor who quotes the heads-up via GitHub's "Quote reply" — which
    preserves HTML comments in the raw markdown — can't trick the
    idempotency check into silently skipping a real heads-up.

    Comments live on the unified issues endpoint regardless of whether the
    item is a PR or an issue, so no ``kind`` argument is required here.
    """
    expected_login = (
        os.environ.get("AGENT_SHIN_BOT_LOGIN") or AGENT_SHIN_DEFAULT_BOT_LOGIN
    ).lower()
    raw = gh(
        "api",
        "--paginate",
        f"repos/{repo}/issues/{number}/comments?per_page=100",
    )
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        comments = payload if isinstance(payload, list) else [payload]
        for comment in comments:
            author = ((comment.get("user") or {}).get("login") or "").lower()
            if author != expected_login:
                continue
            if HEADS_UP_MARKER in (comment.get("body") or ""):
                return True
    return False


def _evaluate_pr(*, repo: str, number: int, model: str, judge: Any = None) -> dict:
    """Run the future PR rubric (review_gate) in dry-run and return the result."""
    return review_gate(
        repo=repo,
        number=number,
        close=False,  # we only want the verdict, never act here
        model=model,
        judge=judge,
    )


def _evaluate_issue(*, repo: str, number: int, model: str, judge: Any = None) -> dict:
    """Run the future issue rubric (triage kind='issue') in dry-run."""
    return triage(
        repo=repo,
        kind="issue",
        number=number,
        close=False,
        model=model,
        judge=judge,
    )


def _would_be_closed(kind: str, result: dict) -> bool:
    """True if the future triage would auto-close this PR/issue based on the
    rubric (regardless of grace-period gating).

    For PRs we trust ``review_gate``'s ``passing`` field — it combines the LLM
    verdict and the Greptile score. For issues we read the LLM verdict
    directly. Both fields are ``None``/missing on skip paths
    (skip-internal-author, skip-llm-error, etc.) where the future bot would
    NOT close the item — those return False.
    """
    if kind == "pr":
        passing = result.get("passing")
        if passing is None:
            return False  # skipped — nothing for the heads-up to warn about
        return passing is False
    verdict = result.get("verdict") or {}
    return (verdict.get("verdict") or "").lower() == "fail"


def _process_one(
    *,
    repo: str,
    kind: str,
    number: int,
    model: str,
    cutoff: dt.date,
    dry_run: bool,
    judge: Any = None,
    skip_marker_check: bool = False,
    allowlist: frozenset[str] = ALLOWLIST_LOGINS,
) -> dict:
    """Evaluate one PR/issue and post a heads-up if it would be auto-closed.

    Returns a per-item dict for the summary table.
    """
    base = {"kind": kind, "number": number}
    fetcher = fetch_pr if kind == "pr" else fetch_issue
    item = fetcher(repo, number)

    if (item.get("state") or "") != "open":
        return {**base, "action": "skip-not-open"}
    if allowlist:
        login = (item.get("user") or {}).get("login") or ""
        if login.lower() not in allowlist:
            return {**base, "action": "skip-not-allowlisted"}
    elif is_internal_contributor(item):
        return {**base, "action": "skip-internal-author"}
    if not skip_marker_check and _has_heads_up_marker(item):
        return {**base, "action": "skip-already-marked-in-body"}
    if not skip_marker_check and _comments_have_marker(repo, number):
        return {**base, "action": "skip-already-notified"}

    if kind == "pr":
        result = _evaluate_pr(repo=repo, number=number, model=model, judge=judge)
    else:
        result = _evaluate_issue(repo=repo, number=number, model=model, judge=judge)

    if not _would_be_closed(kind, result):
        return {**base, "action": "skip-passing", "evaluator": result.get("action")}

    verdict = result.get("verdict") or {}
    greptile_score = result.get("greptile_score") if kind == "pr" else None
    comment = format_heads_up_comment(
        kind=kind, verdict=verdict, greptile_score=greptile_score, cutoff=cutoff
    )
    maybe_post_comment(repo, number, comment, dry_run=dry_run)
    return {
        **base,
        "action": "heads-up-posted" if not dry_run else "would-post-heads-up",
        "verdict": (verdict.get("verdict") or "").lower(),
        "greptile_score": greptile_score,
    }


def _print_summary(results: list[dict]) -> None:
    """Tally per-action counts so a dry-run preview tells you at a glance how
    many comments the real run would post."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    print("\n=== rollout heads-up summary ===")
    for action in sorted(counts):
        print(f"  {action:35s} {counts[action]}")
    print(f"  total                                {len(results)}")


def run(
    *,
    repo: str,
    close: bool,
    cutoff: dt.date,
    model: str,
    kinds: tuple[str, ...] = ("pr", "issue"),
    judge: Any = None,
    only_numbers: dict[str, list[int]] | None = None,
    skip_marker_check: bool = False,
) -> list[dict]:
    """Sweep ``repo`` and post heads-up comments. Returns the per-item results."""
    dry_run = not close
    if dry_run:
        print(
            f"[DRY RUN] sweeping {repo}; --close not passed, no comments will be posted."
        )
    else:
        print(f"[REAL RUN] sweeping {repo}; comments WILL be posted.")
    print(f"Cutoff date in comment body: {cutoff.isoformat()}")

    results: list[dict] = []
    for kind in kinds:
        if only_numbers and kind in only_numbers:
            numbers = list(only_numbers[kind])
        else:
            numbers = _list_open_numbers(repo, kind)
        print(f"\n--- {kind}s: {len(numbers)} open ---")
        for n in numbers:
            try:
                result = _process_one(
                    repo=repo,
                    kind=kind,
                    number=n,
                    model=model,
                    cutoff=cutoff,
                    dry_run=dry_run,
                    judge=judge,
                    skip_marker_check=skip_marker_check,
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - per-item errors don't abort the sweep
                result = {
                    "kind": kind,
                    "number": n,
                    "action": "error",
                    "error": str(exc),
                }
                print(f"!! {kind}#{n}: {exc}", file=sys.stderr)
            print(f"  {kind}#{n}: {result['action']}")
            results.append(result)
    _print_summary(results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument(
        "--close",
        action="store_true",
        help=(
            "Actually post comments. Without this flag the script is in "
            "dry-run mode and only logs what it would do."
        ),
    )
    parser.add_argument(
        "--close-on",
        type=dt.date.fromisoformat,
        default=None,
        help=(
            "Cutoff date shown in the heads-up comment as the rollout date "
            f"(default: today + {DEFAULT_GRACE_DAYS} days)."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL,
        help=f"Model for the rubric LLM judge (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--kind",
        choices=("pr", "issue", "both"),
        default="both",
        help="Restrict the sweep to PRs or issues only (default: both).",
    )
    parser.add_argument(
        "--only-pr",
        type=int,
        action="append",
        default=[],
        help="Limit the PR sweep to these PR numbers (repeat for several).",
    )
    parser.add_argument(
        "--only-issue",
        type=int,
        action="append",
        default=[],
        help="Limit the issue sweep to these issue numbers (repeat for several).",
    )
    parser.add_argument(
        "--ignore-existing-marker",
        action="store_true",
        help=(
            "Re-post on PRs/issues that already carry the heads-up marker. "
            "Useful for testing the comment wording on a known PR."
        ),
    )
    args = parser.parse_args()

    cutoff = args.close_on or (
        dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=DEFAULT_GRACE_DAYS)
    )

    kinds: tuple[str, ...]
    if args.kind == "pr":
        kinds = ("pr",)
    elif args.kind == "issue":
        kinds = ("issue",)
    else:
        kinds = ("pr", "issue")

    only: dict[str, list[int]] = {}
    if args.only_pr:
        only["pr"] = args.only_pr
    if args.only_issue:
        only["issue"] = args.only_issue

    # The script must NOT hit the LLM in dry-run if no key is set — we still
    # want a useful preview that says "skip-no-llm-key" for items that would
    # have been judged. Production runs require OPENAI_API_KEY.
    if args.close and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set for --close (real-run) mode.")

    run(
        repo=args.repo,
        close=args.close,
        cutoff=cutoff,
        model=args.model,
        kinds=kinds,
        only_numbers=only or None,
        skip_marker_check=args.ignore_existing_marker,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
