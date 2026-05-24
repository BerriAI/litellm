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
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Share constants with the sibling Agent Shin script instead of duplicating
# them. `AGENT_SHIN_AUTO_CLOSE_MARKER` is the literal phrase the reconsider
# provenance check keys off, and `INTERNAL_ASSOCIATIONS` is the exempt-author
# set; drift between the two files would silently break reconsider for
# Greptile-closed PRs (marker) or let one script close a PR the other would
# skip (associations), with no test catching it.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from triage_with_llm import (  # noqa: E402
    AGENT_SHIN_AUTO_CLOSE_MARKER,
    INTERNAL_ASSOCIATIONS,
)

# Greptile's GitHub App appears as `greptile-apps[bot]` in REST API comments
# and `greptile-apps` in `gh pr view --json` output. Accept either form.
GREPTILE_BOT_LOGINS = frozenset({"greptile-apps", "greptile-apps[bot]"})

# Matches lines like:
#   <h3>Confidence Score: 3/5</h3>
#   **Confidence Score: 4/5**
#   Confidence Score: 5 / 5
SCORE_PATTERN = re.compile(
    r"confidence\s*score\s*[:\-]?\s*(\d+)\s*/\s*5",
    re.IGNORECASE,
)

# Default labels that exempt a PR from auto-close. Defined at module scope (not
# as a mutable argparse default) so that `--optout-label foo` REPLACES the
# defaults instead of appending to them — the argparse `action="append"` +
# `default=[...]` combination silently mutates the shared default list.
DEFAULT_OPTOUT_LABELS = ("do not close", "keep open", "wip")


def gh(*args: str) -> str:
    """Run a `gh` CLI command and return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


# `gh pr list --limit` caps at 1000 (the CLI's documented hard ceiling).
# Surface a warning if we ever hit that cap so the silent truncation is
# visible in workflow logs instead of just being a missed close.
GH_PR_LIST_LIMIT = 1000


def fetch_open_prs(repo: str | None) -> list[dict]:
    """Fetch all open PRs (number, createdAt, isDraft, labels, author).

    Includes drafts: `gh pr list --state open` returns both ready-for-review
    and draft PRs by default. This is the desired behavior — drafts are not
    a free pass; the internal-collaborator open-PR queue should reflect every
    PR that needs human attention regardless of draft status.
    """
    repo_args = ["--repo", repo] if repo else []
    fields = "number,title,createdAt,isDraft,labels,author,url"
    raw = gh(
        "pr",
        "list",
        "--state",
        "open",
        "--limit",
        str(GH_PR_LIST_LIMIT),
        "--json",
        fields,
        *repo_args,
    )
    prs = json.loads(raw)
    if len(prs) >= GH_PR_LIST_LIMIT:
        # `gh pr list --limit N` returns at most N rows even if more exist;
        # log a GitHub Actions warning so the truncation isn't silent.
        message = (
            f"fetch_open_prs hit the gh CLI cap ({GH_PR_LIST_LIMIT}); "
            "the open-PR list is likely truncated. Switch to paginated "
            "`gh api` calls if the repo regularly exceeds this cap."
        )
        print(f"::warning::{message}", file=sys.stderr)
    return prs


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
    if not association or association in INTERNAL_ASSOCIATIONS:
        return False
    return True


def fetch_pr_comments(pr_number: int, repo: str | None) -> list[dict]:
    """Fetch issue-level comments on a PR (where Greptile posts its summary).

    Returns [] on API failure so a transient hiccup on any single PR doesn't
    abort the whole daily sweep mid-loop. Matches the fail-safe pattern in
    `fetch_pr_author_association`; downstream the empty list becomes a
    `skip-no-greptile-score` action and the PR is re-evaluated on the next run.
    """
    endpoint = (
        f"repos/{repo}/issues/{pr_number}/comments?per_page=100"
        if repo
        else f"repos/{{owner}}/{{repo}}/issues/{pr_number}/comments?per_page=100"
    )
    try:
        raw = gh("api", "--paginate", endpoint)
    except subprocess.CalledProcessError:
        return []
    comments: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            comments.extend(parsed)
        else:
            comments.append(parsed)
    return comments


def extract_greptile_score(comments: Iterable[dict]) -> tuple[int, dict] | None:
    """Return (score, comment) for the most recent Greptile-authored comment
    that contains a "Confidence Score: X/5". Returns None if no such comment.

    "Most recent" is determined by the comment's `updated_at` (falling back to
    `created_at`), so re-reviews override earlier passes.
    """
    candidates: list[tuple[str, int, dict]] = []
    for comment in comments:
        user = (comment.get("user") or {}).get("login", "")
        if user not in GREPTILE_BOT_LOGINS:
            continue
        body = comment.get("body") or ""
        match = SCORE_PATTERN.search(body)
        if not match:
            continue
        score = int(match.group(1))
        timestamp = comment.get("updated_at") or comment.get("created_at") or ""
        candidates.append((timestamp, score, comment))

    if not candidates:
        return None

    candidates.sort(key=lambda triple: triple[0])
    _, score, comment = candidates[-1]
    return score, comment


def parse_iso8601(value: str) -> dt.datetime:
    """Parse a GitHub ISO-8601 timestamp into a timezone-aware datetime."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def has_optout_label(pr: dict, optout_labels: set[str]) -> bool:
    labels = {label.get("name", "").lower() for label in pr.get("labels", [])}
    return bool(labels & {lbl.lower() for lbl in optout_labels})


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

    comment_body = (
        f"👋 Hi, thanks for the PR! {AGENT_SHIN_AUTO_CLOSE_MARKER}, the automated triage "
        "bot for this repository. Closing as part of automated PR triage.\n\n"
        f"Greptile's most recent review scored this PR **{score}/5**, below "
        f"our merge bar of **{threshold}/5**.\n\n"
        "We close low-confidence PRs aggressively to keep the review queue "
        "manageable for maintainers and contributors alike. **This is not a "
        "rejection of the idea** — to bring this back:\n\n"
        "1. Push the fixes that address Greptile's feedback (continue using "
        "your existing branch is fine).\n"
        "2. **Open a new PR** with the updated branch. Greptile will review "
        "it again, and if it scores "
        f"**{threshold}/5 or higher** a maintainer will take another look.\n\n"
        "_Why open a new PR instead of reopening this one?_ GitHub does not "
        "let external contributors reopen a PR that was closed by a bot or "
        "maintainer, so a fresh PR is the most reliable path forward. If you "
        "would prefer this exact PR re-evaluated, comment "
        "`@agent-shin reconsider` once you've pushed the fixes — Agent Shin "
        "will re-run triage and reopen this PR if it now meets the bar.\n\n"
        "Thanks for contributing to LiteLLM. We know auto-closures can sting; "
        "the goal is to keep the project healthy, not to dismiss your work."
    )
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
) -> tuple[str, int | None, int | None]:
    """Decide whether to close `pr`.

    Returns (action, score_or_none, age_days_or_none) where action is one of:
        "skip-too-young", "skip-optout-label", "skip-internal",
        "skip-no-greptile-score", "skip-score-ok", or "close".

    Drafts are NOT skipped — the goal is "open PR count == PRs internal
    collaborators need to action on", and a draft that Greptile scored <4/5
    is still in that queue. Authors can opt out via the `wip` label (see
    `DEFAULT_OPTOUT_LABELS`) if they need to keep a long-lived draft open.
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

    # Only auto-close external OSS contributors. Internal contributors
    # (BerriAI org members) handle their own backlog.
    if not is_external_pr_author(pr, repo):
        return ("skip-internal", None, age_days)

    comments = fetch_pr_comments(pr["number"], repo)
    extraction = extract_greptile_score(comments)
    if extraction is None:
        return ("skip-no-greptile-score", None, age_days)

    score, _ = extraction
    if score >= min_score:
        return ("skip-score-ok", score, age_days)

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
        help=(
            "Maximum number of PRs to close in one run (safety net). "
            "Applied in dry-run too so `--limit N` previews exactly the "
            "first N closures."
        ),
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
        "skip-too-young": 0,
        "skip-optout-label": 0,
        "skip-internal": 0,
        "skip-no-greptile-score": 0,
        "skip-score-ok": 0,
    }

    for pr in sorted(prs, key=lambda p: p["createdAt"]):
        action, score, age_days = evaluate_pr(
            pr,
            now,
            args.min_age_days,
            args.min_score,
            args.repo,
            optout_labels,
        )
        summary[action] = summary.get(action, 0) + 1

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

        closed += 1
        if args.limit is not None and closed >= args.limit:
            verb = "Would close" if dry_run else "Closed"
            print(f"\nReached --limit={args.limit} ({verb} count); stopping.")
            break

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    print(f"\nTotal {'would close' if dry_run else 'closed'}: {summary['close']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
