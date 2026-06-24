#!/usr/bin/env python3
"""
Auto-close low-quality pull requests.

Closes open PRs (including drafts, regardless of age) that satisfy ALL of:
  1. Have a Greptile (`greptile-apps`) review comment whose latest
     "Confidence Score: X/5" is below the configured threshold (default: 4).
  2. Are authored by an external OSS contributor (internal BerriAI
     contributors are exempt).
  3. Do not carry an opt-out label (default: "do not close").

`--min-age-days` is retained as an opt-in safety net for one-off backfill
runs (default: 0). The team's intent is that the count of open PRs equals
the count of PRs internal collaborators need to action on, so neither age
nor draft status acts as a free pass.

For each match, the script posts an explanatory comment and closes the PR.
Because OSS contributors *cannot* reopen a PR closed by the bot/maintainer
(GitHub limitation), the close-comment instructs them to push their fixes
and **open a fresh PR**, or to comment `@agent-shin reconsider` on the
closed PR to have the LLM judge re-evaluate (and reopen on pass).

Requires the `gh` CLI to be authenticated.

Usage examples:
    # Dry run (default) - prints what would be closed
    python3 close_low_quality_prs.py

    # Actually close matching PRs
    python3 close_low_quality_prs.py --close

    # Restrict to PRs at least N days old (one-off backfill safety net)
    python3 close_low_quality_prs.py --min-age-days 7 --min-score 4 --close
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Iterable

# Add this script's directory to `sys.path` so the sibling
# `agent_shin_shared` module is importable when the script is invoked
# directly (e.g. `python3 .github/scripts/close_low_quality_prs.py ...`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_shin_shared import (  # noqa: E402  -- sys.path adjusted above
    AGENT_SHIN_CLOSE_MARKER,
    ALLOWLIST_LOGINS,
    GRACE_COMMENT_MARKER,
    GRACE_PERIOD_SECONDS,
    GREPTILE_BOT_LOGINS,
    SCORE_PATTERN,
    extract_greptile_score,
    gh,
    list_open_items,
    parse_iso8601,
    seconds_since_latest_marker_comment,
)

# `GREPTILE_BOT_LOGINS` and `SCORE_PATTERN` (Greptile's GitHub App login
# variants and the "Confidence Score: X/5" regex) are imported from
# `agent_shin_shared` so the LLM judge in `triage_with_llm.py` and this
# daily Greptile sweep read the score through the same set of logins
# and the same regex.

# `author_association` values for internal BerriAI contributors who should be
# exempt from auto-triage.
INTERNAL_AUTHOR_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# Default labels that exempt a PR from auto-close. Defined at module scope (not
# as a mutable argparse default) so that `--optout-label foo` REPLACES the
# defaults instead of appending to them — the argparse `action="append"` +
# `default=[...]` combination silently mutates the shared default list.
DEFAULT_OPTOUT_LABELS = ("do not close", "keep open", "wip")

# `GRACE_COMMENT_MARKER` (HTML marker appended to grace-period warning
# comments — used by either script to recognize that a warning was
# already posted) and `GRACE_PERIOD_SECONDS` (length of the grace
# period between the warning and the actual auto-close, 2 hours) are
# imported from `agent_shin_shared` so the Agent Shin LLM judge and
# this daily Greptile sweep agree on the same marker and duration.


def fetch_open_prs(repo: str | None) -> list[dict]:
    """Fetch all open PRs (number, createdAt, isDraft, labels, author).

    Includes drafts: `gh pr list --state open` returns both ready-for-review
    and draft PRs by default. This is the desired behavior — drafts are not
    a free pass; the internal-collaborator open-PR queue should reflect every
    PR that needs human attention regardless of draft status.
    """
    fields = "number,title,createdAt,isDraft,labels,author,url"
    return list_open_items("pr", repo=repo, fields=fields)


def fetch_pr_author_association(pr_number: int, repo: str | None) -> str:
    """Return the GitHub `author_association` for a PR, uppercase.

    Values: OWNER, MEMBER, COLLABORATOR, CONTRIBUTOR, FIRST_TIME_CONTRIBUTOR,
    FIRST_TIMER, MANNEQUIN, NONE. Returns "" on lookup failure.
    """
    endpoint = (
        f"repos/{repo}/pulls/{pr_number}"
        if repo
        else f"repos/{{owner}}/{{repo}}/pulls/{pr_number}"
    )
    try:
        data = json.loads(gh("api", endpoint))
    except subprocess.CalledProcessError:
        return ""
    return (data.get("author_association") or "").upper()


def is_external_pr_author(pr: dict, repo: str | None) -> bool:
    """Return True if the PR author is an external OSS contributor.

    Internal = `OWNER` / `MEMBER` / `COLLABORATOR` association, or a bot login.
    """
    login = ((pr.get("author") or {}).get("login") or "").lower()
    if login.endswith("[bot]") or login in {"dependabot", "github-actions"}:
        return False
    association = fetch_pr_author_association(pr["number"], repo)
    # Fail-safe: if the API lookup failed (empty string), treat the author as
    # internal so we don't auto-close their PR. Auto-close is destructive, so
    # an unknown association should never make a PR eligible for closing.
    if not association or association in INTERNAL_AUTHOR_ASSOCIATIONS:
        return False
    return True


def fetch_pr_comments(pr_number: int, repo: str | None) -> list[dict]:
    """Fetch issue-level comments on a PR (where Greptile posts its summary)."""
    endpoint = (
        f"repos/{repo}/issues/{pr_number}/comments?per_page=100"
        if repo
        else f"repos/{{owner}}/{{repo}}/issues/{pr_number}/comments?per_page=100"
    )
    raw = gh("api", "--paginate", endpoint)
    comments: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            # A malformed line should not blow up the whole sweep. Skip and
            # carry on so the remaining PRs in this run still get evaluated.
            continue
        if isinstance(parsed, list):
            comments.extend(parsed)
        else:
            comments.append(parsed)
    return comments


def has_optout_label(pr: dict, optout_labels: set[str]) -> bool:
    labels = {label.get("name", "").lower() for label in pr.get("labels", [])}
    return bool(labels & {lbl.lower() for lbl in optout_labels})


def seconds_since_last_grace_warning(
    comments: Iterable[dict],
    *,
    bot_login: str | None = None,
    now: dt.datetime | None = None,
) -> float | None:
    """Return seconds since the bot's most recent grace-period warning, or
    None if no such warning has ever been posted on this PR.

    Thin wrapper over
    `agent_shin_shared.seconds_since_latest_marker_comment` — the
    centralized helper handles the bot-author filter, marker match,
    timestamp parsing, and `now` injection. Keeping this wrapper
    preserves the closer's "already-fetched comments + injectable now"
    interface so callers (and tests) don't need to change.
    """
    return seconds_since_latest_marker_comment(
        comments,
        marker=GRACE_COMMENT_MARKER,
        bot_login=bot_login,
        now=now,
    )


def format_grace_warning_comment(score: int, threshold: int) -> str:
    """Comment posted on the FIRST low-Greptile-score detection — gives
    the contributor a 2-hour grace window before the auto-close fires on
    the next daily cron run.

    Mirrors `format_grace_warning_pr_comment` in
    `triage_with_llm.py` in spirit (2-hour grace + escape hatches), but
    framed around Greptile's confidence score instead of the LLM judge's
    rubric since the close trigger here is the Greptile signal.
    """
    return (
        "🚅 Hi, thanks for the PR! I'm **Agent Shin**, the automated triage bot for this "
        "repository.\n"
        "\n"
        "Heads up: Greptile's most recent review scored this PR "
        f"**{score}/5**, below our merge bar of **{threshold}/5**.\n"
        "\n"
        "If the score isn't lifted in the next **2 hours**, I'll auto-close this PR. That's "
        "**not** us saying the change isn't worthwhile. We want the open-PR list to mirror "
        "what a maintainer can act on *right now*, so contributors like you don't get lost in "
        "a backlog. Take your time; everything below still works after the close.\n"
        "\n"
        "**During the grace period:** push fixes that address Greptile's feedback, then comment "
        "`@greptileai` to request a fresh review. If "
        f"the new score is **{threshold}/5 or higher**, the PR stays open and no further "
        "action is needed on your side.\n"
        "\n"
        "**If the PR does get auto-closed in 2 hours, you still have an easy recovery path:**\n"
        "\n"
        "- Comment `@greptileai` to request a fresh review. **This still works even after "
        f"the PR is closed**, and a score of {threshold}/5 or higher is one of the signals "
        "that lifts the PR back into the review queue. A low Greptile score isn't a blocker.\n"
        "- Comment `@agent-shin reconsider` after pushing fixes; I'll re-run the rubric and "
        "reopen the PR if both gates (description rubric + Greptile score) now pass.\n"
        "\n"
        f"{GRACE_COMMENT_MARKER}"
    )


def post_grace_warning(
    pr: dict,
    score: int,
    threshold: int,
    repo: str | None,
    dry_run: bool,
) -> None:
    """Post the 2-hour grace-period warning comment on `pr`.

    The warning carries `GRACE_COMMENT_MARKER` so subsequent runs can
    detect that the contributor has already been told about the
    pending close. Does NOT close the PR — the close happens on the
    next eligible run after `GRACE_PERIOD_SECONDS` elapses (handled
    by `close_pr`).
    """
    pr_number = pr["number"]
    repo_args = ["--repo", repo] if repo else []

    if dry_run:
        print(
            f"  [DRY RUN] Would post grace warning to PR #{pr_number} "
            f"(greptile={score}/5): {pr['title']}"
        )
        return

    comment_body = format_grace_warning_comment(score, threshold)
    gh("pr", "comment", str(pr_number), "--body", comment_body, *repo_args)
    print(f"  Posted grace warning on PR #{pr_number} (greptile={score}/5)")


def format_close_comment(score: int, threshold: int) -> str:
    """Comment posted when a low-Greptile-score PR is auto-closed.

    Carries `AGENT_SHIN_CLOSE_MARKER` so the `@agent-shin reconsider` path
    (guarded by `was_closed_by_agent_shin`) recognizes this as an Agent Shin
    close and is allowed to reopen the PR once it passes again; without the
    marker that recovery path the comment advertises silently rejects the
    contributor.
    """
    score_sentence = (
        f"Greptile's most recent review scored this PR **{score}/5**, below "
        f"our merge bar of **{threshold}/5**, and the 2-hour grace period since "
        "the warning has elapsed.\n\n"
    )
    return (
        f"Closing as part of automated PR triage.\n\n"
        f"{score_sentence}"
        "We close low-confidence PRs aggressively to keep the review queue "
        "manageable for maintainers and contributors alike. **This is not a "
        "rejection of the idea.** To bring this back:\n\n"
        "1. Push the fixes that address Greptile's feedback (continue using "
        "your existing branch is fine).\n"
        "2. **Open a new PR** with the updated branch. Greptile will review "
        "it again, and if it scores "
        f"**{threshold}/5 or higher** a maintainer will take another look.\n\n"
        "_Why open a new PR instead of reopening this one?_ GitHub does not "
        "let external contributors reopen a PR that was closed by a bot or "
        "maintainer, so a fresh PR is the most reliable path forward. If you "
        "would prefer this exact PR re-evaluated, comment "
        "`@agent-shin reconsider` once you've pushed the fixes; Agent Shin "
        "will re-run triage and reopen this PR if it now meets the bar. "
        "You can also comment `@greptileai` to request a fresh Greptile "
        "review; that works **even after the PR is closed**.\n\n"
        "Thanks for contributing to LiteLLM. We know auto-closures can sting; "
        "the goal is to keep the project healthy, not to dismiss your work."
        f"\n\n{AGENT_SHIN_CLOSE_MARKER}"
    )


def close_pr(
    pr: dict,
    score: int,
    threshold: int,
    age_days: int,
    repo: str | None,
    dry_run: bool,
    label: str | None,
) -> None:
    """Post the explanatory comment and close the PR."""
    pr_number = pr["number"]
    repo_args = ["--repo", repo] if repo else []

    if dry_run:
        print(
            f"  [DRY RUN] Would close PR #{pr_number} "
            f"(age={age_days}d, greptile={score}/5): {pr['title']}"
        )
        return

    comment_body = format_close_comment(score, threshold)
    gh("pr", "comment", str(pr_number), "--body", comment_body, *repo_args)

    if label:
        try:
            gh("pr", "edit", str(pr_number), "--add-label", label, *repo_args)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            print(f"  warn: failed to add label '{label}' to #{pr_number}: {stderr}")

    gh("pr", "close", str(pr_number), *repo_args)
    print(f"  Closed PR #{pr_number} (greptile={score}/5, age={age_days}d)")


def evaluate_pr(
    pr: dict,
    now: dt.datetime,
    min_age_days: int,
    min_score: int,
    repo: str | None,
    optout_labels: set[str],
    allowlist: frozenset[str] = ALLOWLIST_LOGINS,
) -> tuple[str, int | None, int | None]:
    """Decide what to do with `pr` on this triage run.

    Returns (action, score_or_none, age_days_or_none) where action is one of:
        "skip-too-young", "skip-optout-label", "skip-not-allowlisted",
        "skip-internal", "skip-no-greptile-score", "skip-score-ok",
        "warn-grace", "skip-in-grace-period", or "close".

    Drafts are NOT skipped — the goal is "open PR count == PRs internal
    collaborators need to action on", and a draft that Greptile scored <4/5
    is still in that queue. Authors can opt out via the `wip` label (see
    `DEFAULT_OPTOUT_LABELS`) if they need to keep a long-lived draft open.

    Grace-period semantics: the first time a PR fails the rubric, the
    action is `warn-grace` — the caller should post a warning comment but
    NOT close the PR. On a subsequent run, if the warning is still less
    than `GRACE_PERIOD_SECONDS` old AND the PR still fails, the action is
    `skip-in-grace-period`. Once the warning ages out and the rubric is
    still failing, the action is `close`.
    """
    if has_optout_label(pr, optout_labels):
        return ("skip-optout-label", None, None)

    created = parse_iso8601(pr["createdAt"])
    age_days = (now - created).days
    # `min_age_days` defaults to 0 (close as soon as Greptile scores low).
    # Set a positive value via --min-age-days for one-off backfill runs that
    # want to skip very-young PRs.
    if min_age_days > 0 and age_days < min_age_days:
        return ("skip-too-young", None, age_days)

    # While the allowlist is active it is the sole author gate: only those
    # logins are acted on and the external-only restriction is bypassed for
    # them. Otherwise auto-close only external OSS contributors — internal
    # contributors (BerriAI org members) handle their own backlog.
    login = ((pr.get("author") or {}).get("login") or "").lower()
    if allowlist:
        if login not in allowlist:
            return ("skip-not-allowlisted", None, age_days)
    elif not is_external_pr_author(pr, repo):
        return ("skip-internal", None, age_days)

    comments = fetch_pr_comments(pr["number"], repo)
    extraction = extract_greptile_score(comments)
    if extraction is None:
        return ("skip-no-greptile-score", None, age_days)

    score, _ = extraction
    if score >= min_score:
        return ("skip-score-ok", score, age_days)

    grace_age = seconds_since_last_grace_warning(comments, now=now)
    if grace_age is None:
        return ("warn-grace", score, age_days)
    if grace_age < GRACE_PERIOD_SECONDS:
        return ("skip-in-grace-period", score, age_days)

    return ("close", score, age_days)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository (owner/repo). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=0,
        help=(
            "Minimum age (in days) before a PR is eligible. Default 0 = "
            "close as soon as Greptile flags it. Set a positive value for "
            "one-off backfill runs that want to spare very-young PRs."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=4,
        choices=range(1, 6),
        help="Greptile score below which a PR is closed (default: 4 -> closes <4/5).",
    )
    parser.add_argument(
        "--optout-label",
        action="append",
        default=None,
        help=(
            "Label(s) that exempt a PR from auto-close. Repeat to add more. "
            "Case-insensitive. When omitted, defaults to "
            f"{list(DEFAULT_OPTOUT_LABELS)!r}; passing this flag REPLACES the "
            "defaults (argparse `append` with a mutable default would append "
            "instead, which we explicitly avoid)."
        ),
    )
    parser.add_argument(
        "--close-label",
        type=str,
        default=None,
        help=(
            "Optional label to add to PRs that get auto-closed "
            "(e.g. 'auto-closed-low-quality'). Must already exist on the repo."
        ),
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Actually close matching PRs (default is dry-run).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PRs to close in one run (safety net).",
    )
    args = parser.parse_args()

    dry_run = not args.close
    if dry_run:
        print("=== DRY RUN MODE (pass --close to actually close PRs) ===\n")

    print("Fetching open PRs...")
    prs = fetch_open_prs(args.repo)
    print(f"Found {len(prs)} open PRs.\n")

    now = dt.datetime.now(dt.timezone.utc)
    optout_labels = set(args.optout_label or DEFAULT_OPTOUT_LABELS)

    closed = 0
    summary = {
        "close": 0,
        "warn-grace": 0,
        "skip-in-grace-period": 0,
        "skip-too-young": 0,
        "skip-optout-label": 0,
        "skip-not-allowlisted": 0,
        "skip-internal": 0,
        "skip-no-greptile-score": 0,
        "skip-score-ok": 0,
    }

    # `warned` tracks grace-warning comments posted in this run so the
    # `--limit` safety net bounds *all* destructive write actions, not
    # just closures. Without this cap, a backlog of PRs failing the
    # threshold simultaneously could flood contributors with comments.
    warned = 0
    for pr in sorted(prs, key=lambda p: p["createdAt"]):
        try:
            action, score, age_days = evaluate_pr(
                pr,
                now,
                args.min_age_days,
                args.min_score,
                args.repo,
                optout_labels,
            )
            summary[action] = summary.get(action, 0) + 1

            if action == "warn-grace":
                assert score is not None
                print(
                    f"#{pr['number']}: \"{pr['title']}\" "
                    f"(age={age_days}d, greptile={score}/5) -> warn-grace"
                )
                post_grace_warning(
                    pr,
                    score=score,
                    threshold=args.min_score,
                    repo=args.repo,
                    dry_run=dry_run,
                )
                if not dry_run:
                    warned += 1
                    if args.limit is not None and (warned + closed) >= args.limit:
                        print(
                            f"\nReached --limit={args.limit} "
                            f"(closed={closed}, warned={warned}); stopping."
                        )
                        break
                continue

            if action != "close":
                continue

            assert score is not None and age_days is not None
            print(
                f"#{pr['number']}: \"{pr['title']}\" "
                f"(age={age_days}d, greptile={score}/5) -> close"
            )
            close_pr(
                pr,
                score=score,
                threshold=args.min_score,
                age_days=age_days,
                repo=args.repo,
                dry_run=dry_run,
                label=args.close_label,
            )

            if not dry_run:
                closed += 1
                if args.limit is not None and (warned + closed) >= args.limit:
                    print(
                        f"\nReached --limit={args.limit} "
                        f"(closed={closed}, warned={warned}); stopping."
                    )
                    break
        except Exception as exc:  # noqa: BLE001 - per-PR errors don't abort the sweep
            summary["error"] = summary.get("error", 0) + 1
            print(
                f"!! PR #{pr.get('number')}: {exc}",
                file=sys.stderr,
            )
            continue

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    if dry_run:
        print(f"\nTotal would close: {summary['close']}")
    else:
        print(f"\nTotal closed: {closed}")
    print(
        f"Total {'would warn (grace)' if dry_run else 'warned (grace)'}: "
        f"{summary['warn-grace']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
