"""Constants and helpers shared by Agent Shin's triage scripts.

Both `triage_with_llm.py` (the LLM-judge entrypoint) and
`close_low_quality_prs.py` (the daily Greptile-score sweep) need to
agree on the same notions of:

  * What counts as a Greptile-authored review comment
    (``GREPTILE_BOT_LOGINS``) and how to extract a confidence score from
    its body (``SCORE_PATTERN`` / :func:`extract_greptile_score`).
  * How long the 2-hour grace window is (``GRACE_PERIOD_SECONDS``) and
    the HTML marker stamped into a grace-warning comment so the *other*
    script can see "Agent Shin already warned" and behave accordingly
    (``GRACE_COMMENT_MARKER``).
  * Who Agent Shin is on GitHub (``AGENT_SHIN_DEFAULT_BOT_LOGIN``).
  * How GitHub-style ISO-8601 timestamps round-trip into timezone-aware
    :class:`datetime.datetime` (:func:`parse_iso8601`).

Keeping these in one module means a future change (new Greptile output
format, a longer grace window, a new allowlisted account) is a single edit
instead of two — the original split version had to call out in comments
that the two copies "must stay in sync" precisely because nothing
enforced it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from typing import Iterable

GREPTILE_BOT_LOGINS = frozenset({"greptile-apps", "greptile-apps[bot]"})

SCORE_PATTERN = re.compile(
    r"confidence\s*score\s*[:\-]?\s*(\d+)\s*/\s*5",
    re.IGNORECASE,
)

GRACE_COMMENT_MARKER = "<!-- agent-shin:grace-warning -->"

# Hidden HTML marker stamped on every Agent Shin auto-close comment (the LLM
# judge's grace/review-gate close and the daily Greptile sweep's close).
# `was_closed_by_agent_shin` requires this marker — not just the closing actor —
# before `@agent-shin reconsider` may reopen, because the `github-actions[bot]`
# identity is shared with every other workflow in the repo and is not unique to
# Agent Shin. Both close paths must stamp it or the reconsider path silently
# rejects the contributor.
AGENT_SHIN_CLOSE_MARKER = "<!-- agent-shin:closed -->"

# 2 hours between the grace warning and the auto-close. Short enough to
# dogfood the "fix it before it closes" loop in one sitting; bump back up
# (e.g. 86400 for a day) for the public rollout.
GRACE_PERIOD_SECONDS = 7200

AGENT_SHIN_DEFAULT_BOT_LOGIN = "github-actions[bot]"


def _logins(*names: str) -> frozenset[str]:
    """Build a login set normalized for case-insensitive membership checks.

    Callers compare via ``login.lower() in <set>``, so the stored values
    must be lowercase. Normalizing here lets the literals keep each
    account's canonical GitHub casing (e.g. ``SwiftWinds``) for
    readability without breaking the lookup.
    """
    return frozenset(name.lower() for name in names)


# Dogfood rollout gate. While this set is non-empty, Agent Shin acts ONLY on
# PRs/issues authored by these logins and skips everyone else. For an
# allowlisted author the usual internal/external classification is bypassed, so
# an internal account (e.g. a maintainer's own work login) still gets triaged
# while the bot is being tested on a small set of accounts. Empty the set to
# lift the restriction and restore full triage for the public rollout. Logins
# are compared case-insensitively.
ALLOWLIST_LOGINS = _logins("mateo-berri", "SwiftWinds")

# `gh {pr,issue} list` has no "fetch everything" flag — `--limit` is the only
# control and it defaults to 30. Pass a ceiling far above any realistic open
# backlog (low thousands today) so gh paginates the API until the queue is
# exhausted rather than silently truncating. The bulk sweeps MUST see the whole
# backlog: gh lists newest-first, so a low cap drops the *oldest* PRs/issues —
# exactly the stale ones a low-quality sweep is meant to catch.
GH_LIST_ALL_LIMIT = 100_000


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


def gh(*args: str) -> str:
    """Run a `gh` CLI command and return stdout. Raises on non-zero exit.

    Shared by both Agent Shin entrypoints so a future change here
    (timeout handling, logging, retry on transient failures) only needs
    to be made once.
    """
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def list_open_items(kind: str, *, repo: str | None, fields: str) -> list[dict]:
    """Return EVERY open PR (``kind="pr"``) or issue (``kind="issue"``) in ``repo``.

    Wraps ``gh {pr,issue} list`` with ``--limit GH_LIST_ALL_LIMIT`` so the full
    backlog is fetched instead of the default 30 (or any other arbitrary cap).
    Both bulk sweeps — the daily Greptile closer and the one-shot rollout
    heads-up — rely on this seeing the whole queue, including the oldest items.

    ``fields`` is the comma-separated ``--json`` field list the caller needs
    (e.g. ``"number"`` for the rollout, the full set for the closer).
    """
    if kind not in ("pr", "issue"):
        raise ValueError(f"kind must be 'pr' or 'issue', got {kind!r}")
    repo_args = ["--repo", repo] if repo else []
    raw = gh(
        kind,
        "list",
        "--state",
        "open",
        "--limit",
        str(GH_LIST_ALL_LIMIT),
        "--json",
        fields,
        *repo_args,
    )
    return json.loads(raw)


def seconds_since_latest_marker_comment(
    comments: Iterable[dict],
    *,
    marker: str,
    bot_login: str | None = None,
    now: dt.datetime | None = None,
) -> float | None:
    """Return seconds since the bot's most recent comment containing ``marker``.

    Filters comments by author so a contributor who quotes the HTML
    marker (e.g. via GitHub's "Quote reply" feature, which preserves
    HTML comments in the raw markdown of the quoted text) is not
    mistaken for a bot warning — that would silently reset cooldown
    timers and suppress legitimate notifications.

    ``bot_login`` defaults to the `AGENT_SHIN_BOT_LOGIN` env override or
    ``AGENT_SHIN_DEFAULT_BOT_LOGIN`` so callers normally don't need to
    pass it. ``now`` is injectable for tests / callers (like the daily
    sweep) that want every age calculation pinned to one snapshot.
    """
    expected_login = (
        bot_login
        or os.environ.get("AGENT_SHIN_BOT_LOGIN")
        or AGENT_SHIN_DEFAULT_BOT_LOGIN
    ).lower()
    latest: dt.datetime | None = None
    for comment in comments:
        author = ((comment.get("user") or {}).get("login") or "").lower()
        if author != expected_login:
            continue
        body = comment.get("body") or ""
        if marker not in body:
            continue
        created = comment.get("created_at")
        if not created:
            continue
        try:
            ts = parse_iso8601(created)
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return None
    reference = now if now is not None else dt.datetime.now(dt.timezone.utc)
    return (reference - latest).total_seconds()
