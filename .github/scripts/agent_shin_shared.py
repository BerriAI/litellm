"""Constants and helpers shared by Agent Shin's triage scripts.

Both `triage_with_llm.py` (the LLM-judge entrypoint) and
`close_low_quality_prs.py` (the daily Greptile-score sweep) need to
agree on the same notions of:

  * What counts as a Greptile-authored review comment
    (``GREPTILE_BOT_LOGINS``) and how to extract a confidence score from
    its body (``SCORE_PATTERN`` / :func:`extract_greptile_score`).
  * How long the 1-day grace window is (``GRACE_PERIOD_SECONDS``) and
    the HTML marker stamped into a grace-warning comment so the *other*
    script can see "Agent Shin already warned" and behave accordingly
    (``GRACE_COMMENT_MARKER``).
  * Who Agent Shin is on GitHub (``AGENT_SHIN_DEFAULT_BOT_LOGIN``) and
    which login(s) bypass the grace window entirely
    (``IMMEDIATE_CLOSE_LOGINS``).
  * How GitHub-style ISO-8601 timestamps round-trip into timezone-aware
    :class:`datetime.datetime` (:func:`parse_iso8601`).

Keeping these in one module means a future change (new Greptile output
format, a longer grace window, a new dogfood account) is a single edit
instead of two — the original split version had to call out in comments
that the two copies "must stay in sync" precisely because nothing
enforced it.
"""

from __future__ import annotations

import datetime as dt
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

GRACE_PERIOD_SECONDS = 86400

AGENT_SHIN_DEFAULT_BOT_LOGIN = "github-actions[bot]"

IMMEDIATE_CLOSE_LOGINS = frozenset({"swiftwinds"})


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
