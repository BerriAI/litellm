#!/usr/bin/env python3
"""
Agent Shin — LLM-as-judge triage for external OSS pull requests and issues.

Evaluates a single PR or issue against the contribution rubric and, when the
LLM judge marks it as failing, posts an explanatory comment + closes the
PR/issue. Re-triggers on `reopened` so contributors can iterate back in by
filling in the missing pieces and reopening.

Internal BerriAI contributors (`author_association` in {OWNER, MEMBER,
COLLABORATOR}) and bot accounts are skipped entirely.

Usage:
    triage_with_llm.py --repo owner/repo --pr 1234
    triage_with_llm.py --repo owner/repo --issue 5678
    triage_with_llm.py --repo owner/repo --pr 1234 --close    # actually close
    triage_with_llm.py --repo owner/repo --pr 1234 --print-prompt  # show prompt

Defaults are SAFE: without `--close` the script writes a verdict to stdout (and,
when running in GitHub Actions, to $GITHUB_STEP_SUMMARY) but takes no GitHub
write actions.

Environment:
    GH_TOKEN / GITHUB_TOKEN  - for `gh` CLI auth (auto-set in Actions)
    OPENAI_API_KEY           - required when --close is passed
    OPENAI_BASE_URL          - optional (route to any OpenAI-compatible API)
    TRIAGE_MODEL             - optional model override (default: gpt-5.4-mini)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.parse
from typing import Any, Iterable

# Add this script's directory to `sys.path` so the sibling
# `agent_shin_shared` module is importable when the script is invoked
# directly (e.g. `python3 .github/scripts/triage_with_llm.py ...`) and
# also when the tests load this script via
# `importlib.util.spec_from_file_location`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_shin_shared import (  # noqa: E402  -- sys.path adjusted above
    AGENT_SHIN_CLOSE_MARKER,
    AGENT_SHIN_DEFAULT_BOT_LOGIN,
    ALLOWLIST_LOGINS,
    GRACE_COMMENT_MARKER,
    GRACE_PERIOD_SECONDS,
    GREPTILE_BOT_LOGINS,
    SCORE_PATTERN,
    extract_greptile_score,
    gh,
    parse_iso8601,
    seconds_since_latest_marker_comment,
)

DEFAULT_MODEL = "gpt-5.4-mini"

INTERNAL_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# `AGENT_SHIN_DEFAULT_BOT_LOGIN` is imported from `agent_shin_shared`.
# When the workflow uses the default `secrets.GITHUB_TOKEN`, the
# closure / reopen event's `actor.login` is `github-actions[bot]`. The
# env override `AGENT_SHIN_BOT_LOGIN` exists for local debugging and for
# repos that wire Agent Shin to a PAT.

# HTML marker appended to every reconsider verdict comment. We grep for this
# on subsequent reconsider triggers to enforce a short cooldown so that
# repeated `@agent-shin reconsider` comments don't burn CI/LLM budget.
# Using a unique HTML comment keeps the marker invisible to humans while
# being trivially greppable from a comments-list API response.
RECONSIDER_COMMENT_MARKER = "<!-- agent-shin:reconsider-verdict -->"

# Minimum gap between two reconsider verdicts on the same PR/issue. Set to
# 10 minutes — long enough that a contributor can't trivially spam the
# trigger, short enough that a genuine "I just pushed a fix and reupdated
# the body" iteration loop isn't punished.
RECONSIDER_RATE_LIMIT_SECONDS = 600

# `GRACE_COMMENT_MARKER` (HTML marker on the grace-period warning comment
# posted on the first low-quality detection — used on subsequent triage
# runs to detect that a warning was already posted and measure how long
# ago it was posted) and `GRACE_PERIOD_SECONDS` (length of the grace
# period between the warning and the actual auto-close, 2 hours) are
# imported from `agent_shin_shared` so the daily Greptile sweep and the
# LLM judge agree on the same marker and duration.

# --- Review-gate ("ready for review" label lifecycle) configuration ----------
# The review gate keeps a single label in sync with whether a PR currently
# clears BOTH quality bars: the LLM rubric (clear problem + expected/actual +
# QA proof, or a linked issue) AND Greptile's most recent confidence score.
READY_FOR_REVIEW_LABEL = "ready for review"
DEFAULT_GRACE_DAYS = 1  # 24h before an un-passing, un-tagged PR is auto-closed
DEFAULT_MIN_GREPTILE_SCORE = 4  # Greptile < 4/5 counts as "not passing"

# Hidden HTML-comment markers stamped into review-gate comments. They never
# render in the GitHub UI but let the gate detect its own prior actions so it
# (a) posts the within-grace "what's missing" notice at most once and (b) can
# tell a first-time pass ("ready for review") from a recovery after a
# regression ("all clear again").
READY_MARKER = "<!-- agent-shin:ready -->"
REGRESSED_MARKER = "<!-- agent-shin:regressed -->"
WITHIN_GRACE_MARKER = "<!-- agent-shin:within-grace -->"

# `GREPTILE_BOT_LOGINS` (Greptile's GitHub App login variants —
# `greptile-apps[bot]` in REST API comments, `greptile-apps` in
# `gh pr view --json` output) and `SCORE_PATTERN` (regex matching lines
# like `Confidence Score: 3/5`) are imported from `agent_shin_shared`
# so the daily sweep and the review gate read the score through the
# same set of logins / patterns.

# `AGENT_SHIN_CLOSE_MARKER` is imported from `agent_shin_shared` so this LLM
# judge and the daily Greptile sweep stamp the same marker on their close
# comments — `was_closed_by_agent_shin` keys the reconsider reopen path off it.

# Model families that require `reasoning_effort` to be set, and that reject
# `temperature != 1` unless `reasoning_effort` is "none". For these models we
# pass `reasoning_effort="none"` so a `temperature=0` deterministic judgment
# is still accepted. See litellm/llms/openai/chat/gpt_5_transformation.py for
# the full set of constraints LiteLLM applies to these models.
GPT5_FAMILY_PREFIX = "gpt-5"

# Regexes for picking off "obvious passes" without burning LLM tokens.
#
# Keep this list to GitHub's documented PR-closing keywords only
# (https://docs.github.com/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).
# Casual mentions like "see #1234" or "ref #1234" are intentionally NOT
# auto-passed — they should fall through to the LLM judge, which has the
# stricter rubric "a bare issue number without a closing keyword counts only
# if it's clearly the related issue (not a passing mention)".
LINKED_ISSUE_PATTERN = re.compile(
    r"\b(?:fixes|fix|fixed|closes|close|closed|resolves|resolve|resolved)\s+"
    r"(?:#\d+|https?://github\.com/[\w.-]+/[\w.-]+/issues/\d+)",
    re.IGNORECASE,
)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


# ---------------------------------------------------------------------------
# gh helpers
#
# `gh` is imported from `agent_shin_shared` so a future change (timeout,
# logging, retry) only needs to be made once.


def fetch_pr(repo: str, number: int) -> dict:
    """Return the full GitHub REST representation of a PR."""
    return json.loads(gh("api", f"repos/{repo}/pulls/{number}"))


def fetch_issue(repo: str, number: int) -> dict:
    """Return the full GitHub REST representation of an issue."""
    return json.loads(gh("api", f"repos/{repo}/issues/{number}"))


def post_comment(repo: str, number: int, body: str) -> None:
    """Post an issue-style comment (works for both issues and PRs)."""
    gh(
        "api",
        f"repos/{repo}/issues/{number}/comments",
        "-X",
        "POST",
        "-f",
        f"body={body}",
    )


def close_pr(repo: str, number: int) -> None:
    """Close a pull request (state=closed)."""
    gh(
        "api",
        f"repos/{repo}/pulls/{number}",
        "-X",
        "PATCH",
        "-f",
        "state=closed",
    )


def reopen_pr(repo: str, number: int) -> None:
    """Reopen a previously-closed pull request (state=open).

    Used by the `@agent-shin reconsider` comment-trigger flow: the bot has
    write access via GH_TOKEN, so it can reopen on the contributor's behalf
    even though GitHub doesn't let the OSS author do it themselves.
    """
    gh(
        "api",
        f"repos/{repo}/pulls/{number}",
        "-X",
        "PATCH",
        "-f",
        "state=open",
    )


def close_issue(repo: str, number: int, *, not_planned: bool = True) -> None:
    """Close an issue, marking state_reason=not_planned by default."""
    args = [
        "api",
        f"repos/{repo}/issues/{number}",
        "-X",
        "PATCH",
        "-f",
        "state=closed",
    ]
    if not_planned:
        args.extend(["-f", "state_reason=not_planned"])
    gh(*args)


def reopen_issue(repo: str, number: int) -> None:
    """Reopen a previously-closed issue (state=open, state_reason=reopened)."""
    gh(
        "api",
        f"repos/{repo}/issues/{number}",
        "-X",
        "PATCH",
        "-f",
        "state=open",
        "-f",
        "state_reason=reopened",
    )


def add_label(repo: str, number: int, label: str) -> None:
    """Add a label to a PR/issue (GitHub creates the label if it's missing)."""
    gh(
        "api",
        f"repos/{repo}/issues/{number}/labels",
        "-X",
        "POST",
        "-f",
        f"labels[]={label}",
    )


def remove_label(repo: str, number: int, label: str) -> None:
    """Remove a label from a PR/issue. A missing label (404) is not an error."""
    encoded = urllib.parse.quote(label, safe="")
    try:
        gh(
            "api",
            f"repos/{repo}/issues/{number}/labels/{encoded}",
            "-X",
            "DELETE",
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").lower()
        if "404" in stderr or "not found" in stderr:
            return
        raise


def _iter_paginated_json(*api_args: str) -> Any:
    """Yield JSON objects from `gh api --paginate ... -q '.[]'`.

    `gh api --paginate` on a JSON-array endpoint concatenates pages into
    one stream; `-q '.[]'` flattens that stream into newline-delimited
    objects (jq-style). This keeps memory bounded for chatty endpoints
    like issue events/comments on long-lived PRs.
    """
    raw = gh("api", "--paginate", *api_args, "-q", ".[]")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # A malformed line should not blow up the whole guard. Skip and
            # carry on — at worst the guard fail-closes (returns False /
            # None) and the caller treats it as "unknown".
            continue


def fetch_last_close_event(
    repo: str, number: int
) -> tuple[str | None, dt.datetime | None]:
    """Return the actor login and timestamp of the most recent `closed` event.

    Either field may be None: actor when the events API returns nothing
    (unusual for a closed item, but possible on transient errors), and
    timestamp when the event lacks `created_at` or the value can't be
    parsed. `was_closed_by_agent_shin` fail-closes on either.
    """
    actor: str | None = None
    closed_at: dt.datetime | None = None
    for event in _iter_paginated_json(f"repos/{repo}/issues/{number}/events"):
        if event.get("event") != "closed":
            continue
        actor = (event.get("actor") or {}).get("login")
        created = event.get("created_at")
        if not created:
            closed_at = None
            continue
        try:
            closed_at = parse_iso8601(created)
        except ValueError:
            closed_at = None
    return actor, closed_at


# How much older than the latest `closed` event the Agent Shin marker
# comment is allowed to be while still counting as "this close was Agent
# Shin's". Agent Shin posts the close comment immediately before closing,
# so the marker timestamp is normally at most a few seconds before the
# close event; the buffer just absorbs clock skew between the comments
# API and the events API.
AGENT_SHIN_CLOSE_MARKER_SKEW_SECONDS = 300


def was_closed_by_agent_shin(
    repo: str, number: int, *, bot_login: str | None = None
) -> bool:
    """Return True iff Agent Shin itself most-recently closed this PR/issue.

    This is the guard that stops `@agent-shin reconsider` from reopening an
    item Agent Shin did not close — a maintainer closing for non-rubric
    reasons (security, duplicate, design rejection), or a different workflow
    (stale/duplicate sweeps) closing under the shared `github-actions[bot]`
    identity. Three independent signals must all hold, because that identity
    is not unique to Agent Shin and a marker comment from a prior
    closed/reopened cycle would otherwise vouch for an unrelated close:

      1. The most recent `closed` event's actor is the bot identity.
      2. Agent Shin left one of its auto-close comments, detected via
         `AGENT_SHIN_CLOSE_MARKER`. The actor check alone can't tell an
         Agent Shin close from any other `github-actions[bot]` close.
      3. That marker comment was posted at (or just before) the latest
         close event, not on a previous close in an
         Agent-Shin-close -> reconsider-reopen -> other-bot-reclose cycle.

    The check is intentionally fail-closed: any uncertainty about who closed
    the item is treated as "not Agent Shin" so the destructive reopen path
    stays gated.
    """
    expected = (
        bot_login
        or os.environ.get("AGENT_SHIN_BOT_LOGIN")
        or AGENT_SHIN_DEFAULT_BOT_LOGIN
    ).lower()
    actor, closed_at = fetch_last_close_event(repo, number)
    if not actor or actor.lower() != expected or closed_at is None:
        return False
    marker_seconds = seconds_since_last_agent_shin_close(
        repo, number, bot_login=bot_login
    )
    if marker_seconds is None:
        return False
    close_age_seconds = (dt.datetime.now(dt.timezone.utc) - closed_at).total_seconds()
    return marker_seconds <= close_age_seconds + AGENT_SHIN_CLOSE_MARKER_SKEW_SECONDS


def _seconds_since_latest_marker_comment(
    repo: str,
    number: int,
    *,
    marker: str,
    bot_login: str | None = None,
) -> float | None:
    """Return seconds since the bot's most recent comment with ``marker``.

    Fetches comments via `_iter_paginated_json` and delegates the
    iteration / author-filter / timestamp logic to
    `agent_shin_shared.seconds_since_latest_marker_comment` so the daily
    Greptile sweep and the LLM judge use one source of truth for the
    "bot already posted X" detection. The wall-clock `now` is resolved
    against this module's `dt` so tests that freeze time via
    `monkeypatch.setattr(triage_module, "dt", ...)` still apply.
    """
    return seconds_since_latest_marker_comment(
        _iter_paginated_json(f"repos/{repo}/issues/{number}/comments"),
        marker=marker,
        bot_login=bot_login,
        now=dt.datetime.now(dt.timezone.utc),
    )


def seconds_since_last_reconsider_verdict(
    repo: str, number: int, *, bot_login: str | None = None
) -> float | None:
    """Return seconds since the bot's most recent reconsider verdict comment.

    Detects comments by matching the HTML marker `RECONSIDER_COMMENT_MARKER`
    appended by `format_reopen_comment` and
    `format_reconsider_still_failing_comment`. Returns None when the bot
    has never posted a reconsider verdict on this PR/issue (or when the
    only matching comments are missing a `created_at` timestamp, which
    shouldn't happen on a real GitHub response).
    """
    return _seconds_since_latest_marker_comment(
        repo, number, marker=RECONSIDER_COMMENT_MARKER, bot_login=bot_login
    )


def seconds_since_last_grace_warning(
    repo: str, number: int, *, bot_login: str | None = None
) -> float | None:
    """Return seconds since the bot's most recent grace-period warning.

    Detects warning comments by matching the HTML marker
    `GRACE_COMMENT_MARKER` appended by `format_grace_warning_pr_comment`
    and `format_grace_warning_issue_comment`. Returns None when no
    grace warning has ever been posted on this PR/issue — that's the
    "first low-quality detection" signal that drives the warning path.
    """
    return _seconds_since_latest_marker_comment(
        repo, number, marker=GRACE_COMMENT_MARKER, bot_login=bot_login
    )


def seconds_since_last_agent_shin_close(
    repo: str, number: int, *, bot_login: str | None = None
) -> float | None:
    """Return seconds since Agent Shin's most recent auto-close comment.

    Detects close comments by matching `AGENT_SHIN_CLOSE_MARKER` (stamped by
    `format_pr_close_comment` / `format_issue_close_comment`). Returns None
    when Agent Shin has never closed this PR/issue — the signal
    `was_closed_by_agent_shin` uses to keep the reconsider reopen path gated
    against closures performed by other workflows sharing the bot identity.
    """
    return _seconds_since_latest_marker_comment(
        repo, number, marker=AGENT_SHIN_CLOSE_MARKER, bot_login=bot_login
    )


# ---------------------------------------------------------------------------
# Author classification


def is_internal_contributor(item: dict) -> bool:
    """Return True if the PR/issue author should be exempted from triage.

    Fail-safe: if `author_association` is missing or empty (which should never
    happen on a successful GitHub REST response but is possible on schema
    changes or partial responses), treat the author as INTERNAL so the
    destructive close path never fires on an unknown contributor. This matches
    the sibling `is_external_pr_author` in `close_low_quality_prs.py`.
    """
    login = ((item.get("user") or {}).get("login") or "").lower()
    if login.endswith("[bot]") or login in {"dependabot", "github-actions"}:
        return True
    association = (item.get("author_association") or "").upper()
    if not association or association in INTERNAL_ASSOCIATIONS:
        return True
    return False


# ---------------------------------------------------------------------------
# Greptile score + age helpers (`extract_greptile_score`, `parse_iso8601`)
# live in `agent_shin_shared` — they're imported at the top of this module
# so both `triage_with_llm.py` and `close_low_quality_prs.py` share a
# single source of truth for the Confidence-Score regex and ISO-8601
# parsing.


# ---------------------------------------------------------------------------
# Prompt construction


def strip_html_comments(text: str) -> str:
    """Remove HTML comments — template placeholder text shouldn't fool the judge."""
    return HTML_COMMENT_PATTERN.sub("", text or "")


def has_linked_issue(text: str) -> bool:
    """Heuristic: does this body link to an open issue (Fixes #123 etc.)?"""
    return bool(LINKED_ISSUE_PATTERN.search(strip_html_comments(text or "")))


def build_pr_prompt(*, title: str, body: str) -> str:
    cleaned_body = strip_html_comments(body or "").strip() or "(empty)"
    # Dedent the static template *before* interpolating dynamic fields so that
    # multi-line bodies (whose 2nd+ lines start at column 0) don't defeat the
    # common-indent computation in textwrap.dedent.
    template = textwrap.dedent("""
        You are "Agent Shin", the OSS triage bot for the LiteLLM open-source
        repository (BerriAI/litellm). Decide whether this external pull request
        meets the project's contribution standards.

        A PR PASSES triage only if BOTH (1) AND (2) are satisfied. A linked
        issue alone is NOT enough — it covers context, not proof.

          (1) CONTEXT — the PR provides AT LEAST ONE of:
              (a) A link to a related GitHub issue. Acceptable forms:
                  "Fixes #1234", "Closes #1234", "Resolves #1234",
                  "Refs https://github.com/BerriAI/litellm/issues/1234". A
                  bare "#1234" without a closing keyword counts only if it
                  is clearly the related issue (not a passing mention).
              (b) A clear problem description in the body (what bug or
                  missing feature this addresses, beyond the title) AND
                  expected vs. actual behavior (or, for features, "what's
                  possible now vs. with this PR").

          (2) END-TO-END QA PROOF: the PR body contains AT LEAST ONE of:
              (a) A screen recording / video showing the behavior before
                  and after the change (the bug reproducing, then the fix
                  working). For a brand-new feature with no meaningful
                  "before", a recording of it working end-to-end is fine.
              (b) A screenshot (or before/after screenshots) showing the
                  fix or feature working.
              (c) Specific commands that were actually run (curl, python,
                  a CLI invocation, etc.) PAIRED WITH their real
                  output, demonstrating the change works end-to-end against
                  the real system. Commands whose external dependencies
                  (LLM provider, DB, network) are mocked or stubbed do NOT
                  satisfy (2c); they are not end-to-end.

              `has_qa_proof` must be set to `true` only when (2a), (2b),
              or a non-mocked (2c) is actually present in the body. If the
              only "proof" is mocked tests, `has_qa_proof` is `false` and
              the verdict is "fail".

              The following do NOT count as QA proof:
              - Generic claims like "I tested it", "works locally", "all
                tests pass", or a checked "I added tests" checkbox with no
                output shown.
              - A description of what tests exist or were added, without
                their actual output in the PR body.
              - `pytest` (or any test runner) executed against the
                repository's own unit tests. Those mock the LLM provider,
                DB, and network, so they are NOT end-to-end and never
                satisfy (2), no matter how much passing output is pasted.
              - A linked issue. The linked issue is context (1a), never
                proof (2).

        FAIL the PR if EITHER (1) or (2) is missing. Do not bias toward PASS:
        if QA proof is absent, the verdict is "fail" even when the rest of
        the PR is well-written.

        Respond with a single JSON object, no prose:

        {{
          "verdict": "pass" | "fail",
          "linked_issue": boolean,
          "has_problem_description": boolean,
          "has_expected_vs_actual": boolean,
          "has_qa_proof": boolean,
          "qa_proof_type": "video" | "screenshot" | "commands_with_output" | "none",
          "missing": ["plain-english strings naming what is missing"],
          "explanation": "1-2 sentence reasoning for the team to skim"
        }}

        ---
        PR title: {title}

        PR body:
        ---
        {cleaned_body}
        ---
        """).strip()
    return template.format(title=title, cleaned_body=cleaned_body)


def build_issue_prompt(*, title: str, body: str) -> str:
    cleaned_body = strip_html_comments(body or "").strip() or "(empty)"
    # Dedent the static template *before* interpolating dynamic fields so that
    # multi-line bodies (whose 2nd+ lines start at column 0) don't defeat the
    # common-indent computation in textwrap.dedent.
    template = textwrap.dedent("""
        You are "Agent Shin", the OSS triage bot for the LiteLLM open-source
        repository (BerriAI/litellm). Decide whether this GitHub issue meets
        the project's reporting standards.

        For a BUG REPORT the issue PASSES triage only when it contains BOTH:
          (1) END-TO-END EVIDENCE OF THE BUG (the "before"; set
              `has_repro=true` only when this is present): AT LEAST ONE of:
              (a) A screen recording / video of the bug happening.
              (b) A screenshot of the bug.
              (c) The exact command(s) actually run (curl, python, a CLI
                  invocation, etc.) PAIRED WITH their real output, traceback,
                  or logs showing the failure against the real system.
                  Commands whose external dependencies (LLM provider, DB,
                  network) are mocked or stubbed do NOT count.
              Prose-only "steps to reproduce" with no run output, video, or
              screenshot do NOT satisfy (1).
          (2) Expected vs. actual behavior (`has_expected_vs_actual`).

        FAIL the bug report if either (1) or (2) is missing. Do not bias
        toward PASS: if the bug isn't demonstrated end-to-end, the verdict is
        "fail" even when the report is well-written.

        For a FEATURE REQUEST the issue PASSES triage only when it contains
        ALL of:
          - A clear description of the proposed feature (what should LiteLLM do
            that it does not today).
          - Motivation / use case with a concrete example (config, API call,
            UI flow, or scenario showing what's blocked today).

        For an issue that is neither a bug report nor a feature request (a
        question, support request, or discussion), PASS as long as it has a
        clear, specific ask and is not empty or template placeholder text.

        Respond with a single JSON object, no prose:

        {{
          "verdict": "pass" | "fail",
          "kind": "bug" | "feature" | "other",
          "has_repro": boolean,
          "has_expected_vs_actual": boolean,
          "has_motivation_example": boolean,
          "missing": ["plain-english strings naming what is missing"],
          "explanation": "1-2 sentence reasoning for the team to skim"
        }}

        ---
        Issue title: {title}

        Issue body:
        ---
        {cleaned_body}
        ---
        """).strip()
    return template.format(title=title, cleaned_body=cleaned_body)


# ---------------------------------------------------------------------------
# LLM call + verdict parsing


def call_llm_judge(
    prompt: str, *, model: str, api_key: str, base_url: str | None
) -> str:
    """Call an OpenAI-compatible chat completions endpoint. Returns raw text."""
    # Import inside the function so unit tests that monkey-patch this never
    # need the openai package installed.
    from openai import OpenAI

    client = (
        OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else OpenAI(api_key=api_key)
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    # gpt-5.x reasoning models reject `temperature != 1` unless
    # `reasoning_effort` is explicitly "none". Set it via `extra_body` so this
    # works across openai SDK versions regardless of whether the SDK natively
    # types `reasoning_effort` as a top-level chat-completions param yet.
    if model.lower().startswith(GPT5_FAMILY_PREFIX):
        kwargs["extra_body"] = {"reasoning_effort": "none"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def parse_verdict(raw: str) -> dict:
    """Parse the LLM's JSON response. Tolerates ```json fences and stray text."""
    if not raw:
        raise ValueError("empty LLM response")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"could not extract JSON from LLM response: {raw[:200]}")
        return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Comment composition


def _format_missing(missing: list[str]) -> str:
    if not missing:
        return "- (see explanation below)"
    return "\n".join(f"- {m}" for m in missing)


# Rubric items the judge can mark present. The first element of each tuple is
# the verdict-JSON boolean field, the second is the human-readable label we
# render in the "what you got right" section of close / grace-warning comments.
_PR_PRESENT_LABELS: tuple[tuple[str, str], ...] = (
    ("linked_issue", "Linked a related GitHub issue"),
    ("has_problem_description", "Clear problem description"),
    ("has_expected_vs_actual", "Expected vs. actual behavior"),
    ("has_qa_proof", "End-to-end QA proof"),
)

# Issue rubric labels grouped by `kind`. The judge sets `kind` to one of
# {"bug", "feature", "other"}; when "other" we render both groups so we don't
# silently drop a present-flag the judge actually set to True.
_ISSUE_BUG_LABELS: tuple[tuple[str, str], ...] = (
    (
        "has_repro",
        "End-to-end evidence of the bug (video, screenshot, or command + real output)",
    ),
    ("has_expected_vs_actual", "Expected vs. actual behavior"),
)
_ISSUE_FEATURE_LABELS: tuple[tuple[str, str], ...] = (
    ("has_motivation_example", "Motivation and concrete example"),
)


def _format_present_for_pr(verdict: dict) -> list[str]:
    """Human-readable rubric items the judge confirmed are present on a PR.

    Drives the "what you got right" section in close / grace-warning comments.
    The user gave explicit feedback: contributors should see what they nailed
    *before* the list of gaps, so the comment doesn't read as pure rejection.
    """
    return [label for field, label in _PR_PRESENT_LABELS if verdict.get(field)]


def _format_present_for_issue(verdict: dict) -> list[str]:
    """Human-readable rubric items the judge confirmed are present on an issue.

    Branches on the judge's `kind` field. For `"other"` (or missing kind) we
    render the union so a present-flag isn't dropped just because the judge
    couldn't classify the issue cleanly.
    """
    kind = (verdict.get("kind") or "").lower()
    groups: list[tuple[tuple[str, str], ...]] = []
    if kind in ("bug", "other", ""):
        groups.append(_ISSUE_BUG_LABELS)
    if kind in ("feature", "other", ""):
        groups.append(_ISSUE_FEATURE_LABELS)
    out: list[str] = []
    for group in groups:
        for field, label in group:
            if verdict.get(field) and label not in out:
                out.append(label)
    return out


def _format_present_block(items: list[str]) -> str:
    """Render the optional "what you got right" block. Empty string when the
    judge didn't confirm anything as present — better to omit the section
    entirely than to show "What you got right: (nothing)".
    """
    if not items:
        return ""
    bullets = "\n".join(f"- ✅ {item}" for item in items)
    return f"**What you got right:**\n\n{bullets}\n\n"


def format_pr_close_comment(verdict: dict) -> str:
    missing_lines = _format_missing(verdict.get("missing") or [])
    present_block = _format_present_block(_format_present_for_pr(verdict))
    explanation = verdict.get("explanation") or ""
    return (
        "🚅 Hi, thanks for the PR! I'm **Agent Shin**, the automated triage bot for this "
        "repository. "
        "[What's this and why am I getting it?](https://docs.litellm.ai/blog/agent-shin-triage)\n"
        "\n"
        "I read the description against our "
        "[contribution rubric](https://github.com/BerriAI/litellm/blob/main/.github/pull_request_template.md). "
        "Here's how it lined up:\n"
        "\n"
        f"{present_block}"
        "**What's still missing:**\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "**Closing this PR isn't a rejection of the change.** We want the open-PR list to "
        "mirror what a maintainer can act on *right now*, so contributors don't get lost in a "
        'backlog. A closed PR is a soft "park this for later"; your work is still here, '
        "the diff is still here, and getting it reopened is one comment away. Take your time.\n"
        "\n"
        "**To bring this PR back:**\n"
        "\n"
        "- Update the description with the missing pieces, then comment `@agent-shin reconsider` "
        "on this PR. I'll re-evaluate and reopen if it now passes.\n"
        "- Or **Open a new PR** with the same fix and the updated description. GitHub doesn't "
        "always let external contributors reopen a bot-closed PR, so a fresh PR is the most "
        "reliable path back into the review queue.\n"
        "- If Greptile's most recent score on this PR was below 4/5, comment `@greptileai` to "
        "request a fresh review; that **still works even after the PR is closed**, and a "
        "stronger score is one of the signals that lifts the PR back into the queue. A low "
        "Greptile score isn't a blocker.\n"
        "\n"
        '**What "end-to-end QA proof" means**, since it\'s the most common gap: at least one '
        "of a short before/after screen recording / video (the bug reproducing, then the fix "
        "working; for a brand-new feature, a recording of it working end-to-end), a screenshot "
        "(or before/after screenshots) of it working, or the exact commands you ran paired "
        "with their **real output** against the real system. Running `pytest` on the repo's "
        "unit tests doesn't count; those mock the LLM provider, DB, and network, so they "
        "aren't end-to-end. Output from a real, no-mocks integration run is what we look "
        "for. A linked issue alone isn't enough either: it covers context, not proof. See "
        "[the full rubric](https://docs.litellm.ai/blog/agent-shin-triage#the-rubric-for-pull-requests).\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you; ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, comment "
        "`@agent-shin reconsider` or ping a maintainer; they'll override me.)_"
        f"\n\n{AGENT_SHIN_CLOSE_MARKER}"
    )


def format_issue_close_comment(verdict: dict) -> str:
    missing_lines = _format_missing(verdict.get("missing") or [])
    present_block = _format_present_block(_format_present_for_issue(verdict))
    explanation = verdict.get("explanation") or ""
    return (
        "🚅 Hi, thanks for filing this! I'm **Agent Shin**, the automated triage bot for this "
        "repository. "
        "[What's this and why am I getting it?](https://docs.litellm.ai/blog/agent-shin-triage)\n"
        "\n"
        "I read the issue against our reporting checklist. Here's how it lined up:\n"
        "\n"
        f"{present_block}"
        "**What's still missing:**\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "**Closing this isn't us saying the bug isn't real or the request isn't useful.** We "
        "want the open-issue list to mirror what a maintainer can act on *right now*, so "
        "reports like yours don't get buried in a backlog. A closed issue is a soft \"park "
        'this for later"; your report is still here, and getting it reopened is one comment '
        "away. Take your time.\n"
        "\n"
        "**To bring this issue back:**\n"
        "\n"
        "1. Edit the issue description to add the missing pieces:\n"
        "   - For **bug reports**: end-to-end evidence of the bug (a screen recording / "
        "video, a screenshot, or the exact commands you ran with their real output / "
        "traceback) plus expected vs. actual behavior. Written steps with no run output, "
        "video, or screenshot don't count, and mocked or stubbed runs don't count.\n"
        "   - For **feature requests**: a concrete description of what should change, plus a "
        "use case and example (config / API call / UI flow).\n"
        "2. Comment `@agent-shin reconsider`. I'll re-run triage and reopen the issue if it "
        "now meets the bar. (GitHub doesn't let external authors reopen an issue a maintainer "
        "or bot closed, so the comment-based reconsider is the reliable path.)\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you; ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, comment "
        "`@agent-shin reconsider` or ping a maintainer; they'll override me.)_"
        f"\n\n{AGENT_SHIN_CLOSE_MARKER}"
    )


def format_grace_warning_pr_comment(verdict: dict) -> str:
    """Comment posted on the FIRST low-quality detection — gives the
    contributor a 2-hour grace window to fix the PR before the next
    triage run actually closes it.

    This is the "before-close" warning. On the second triage run, if the
    grace marker is older than `GRACE_PERIOD_SECONDS` AND the PR still
    fails the rubric, the close path runs (which posts
    `format_pr_close_comment` and closes the PR).
    """
    missing_lines = _format_missing(verdict.get("missing") or [])
    present_block = _format_present_block(_format_present_for_pr(verdict))
    explanation = verdict.get("explanation") or ""
    return (
        "🚅 Hi, thanks for the PR! I'm **Agent Shin**, the automated triage bot for this "
        "repository. "
        "[What's this and why am I getting it?](https://docs.litellm.ai/blog/agent-shin-triage)\n"
        "\n"
        "I read the description against our "
        "[contribution rubric](https://github.com/BerriAI/litellm/blob/main/.github/pull_request_template.md). "
        "Here's how it lined up:\n"
        "\n"
        f"{present_block}"
        "**What's still missing:**\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "If the description isn't updated in the next **2 hours**, I'll auto-close this PR. "
        "That's **not** us saying we don't care about the change; we want the open-PR list to "
        "mirror what a maintainer can act on *right now*, so contributors don't get lost in a "
        'backlog. A closed PR is a soft "park this for later," not a rejection. Take your '
        "time; everything below still works after the close.\n"
        "\n"
        "**During the grace period:** just update the PR description with the missing pieces. "
        "No need to ping me; I'll re-check on the next sweep and skip the auto-close if it "
        "now passes. See "
        "[what counts as QA proof](https://docs.litellm.ai/blog/agent-shin-triage#the-rubric-for-pull-requests) "
        "for the full rubric (a linked issue alone isn't enough; it covers context, not proof).\n"
        "\n"
        "**If the PR does get auto-closed in 2 hours, you still have easy recovery paths:**\n"
        "\n"
        "- Comment `@agent-shin reconsider` after updating the description. I'll re-evaluate "
        "and reopen the PR if it now passes.\n"
        "- Comment `@greptileai` to request a fresh Greptile review; that **still works even "
        "after the PR is closed**, and a stronger score is one of the signals that lifts the "
        "PR back into the queue. So a low Greptile score isn't a blocker either.\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you; ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, ping a "
        "maintainer; they'll override me.)_\n"
        "\n"
        f"{GRACE_COMMENT_MARKER}"
    )


def format_grace_warning_issue_comment(verdict: dict) -> str:
    """Issue analogue of `format_grace_warning_pr_comment`."""
    missing_lines = _format_missing(verdict.get("missing") or [])
    present_block = _format_present_block(_format_present_for_issue(verdict))
    explanation = verdict.get("explanation") or ""
    return (
        "🚅 Hi, thanks for filing this! I'm **Agent Shin**, the automated triage bot for this "
        "repository. "
        "[What's this and why am I getting it?](https://docs.litellm.ai/blog/agent-shin-triage)\n"
        "\n"
        "I read the issue against our reporting checklist. Here's how it lined up:\n"
        "\n"
        f"{present_block}"
        "**What's still missing:**\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "If the issue isn't updated in the next **2 hours**, I'll auto-close it. That's **not** us "
        "saying the bug isn't real or the request isn't useful; we want the open-issue list "
        "to mirror what a maintainer can act on *right now*, so reports like yours don't get "
        'buried in a backlog. A closed issue is a soft "park this for later," not a '
        "rejection. Take your time; reopening is one comment away.\n"
        "\n"
        "**During the grace period:** just edit the issue description with the missing "
        "pieces. No need to ping me; I'll re-check on the next sweep and skip the auto-close "
        "if it now passes.\n"
        "\n"
        "Missing pieces, depending on what this is:\n"
        "\n"
        "- For **bug reports**: end-to-end evidence of the bug (a screen recording / video, a "
        "screenshot, or the exact commands you ran with their real output / traceback) plus "
        "expected vs. actual behavior. Written steps with no run output don't count, and "
        "mocked or stubbed runs don't count.\n"
        "- For **feature requests**: a concrete description of what should change, plus a use "
        "case and example (config / API call / UI flow).\n"
        "\n"
        "**If the issue does get auto-closed in 2 hours**, comment `@agent-shin reconsider` "
        "and I'll re-evaluate. If it now meets the bar, I'll reopen the issue.\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you; ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, ping a "
        "maintainer; they'll override me.)_\n"
        "\n"
        f"{GRACE_COMMENT_MARKER}"
    )


# ---------------------------------------------------------------------------
# Step-summary helpers


def write_step_summary(content: str) -> None:
    """When running inside GitHub Actions, append to the step summary file."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        print(f"warn: failed to write step summary: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core orchestration


def format_reopen_comment(kind: str) -> str:
    """Comment posted when Agent Shin reopens after a successful reconsider."""
    noun = "PR" if kind == "pr" else "issue"
    # The trailing HTML marker is used by `seconds_since_last_reconsider_verdict`
    # to enforce a cooldown between repeated `@agent-shin reconsider` triggers.
    # Keep the marker on its own line so it doesn't disturb the rendered text.
    return (
        f"♻️ **Re-evaluated and reopened.** Thanks for updating the {noun}!\n"
        "\n"
        "Agent Shin re-ran triage on the latest description and it now meets "
        "the bar. A maintainer will take another look soon; please don't "
        f"close this {noun} again unless asked to.\n"
        "\n"
        "_(If a maintainer ends up closing this for non-rubric reasons, that "
        "decision stands; comment `@agent-shin reconsider` again only if you "
        "have substantively new information.)_\n"
        "\n"
        f"{RECONSIDER_COMMENT_MARKER}"
    )


def format_reconsider_still_failing_comment(kind: str, verdict: dict) -> str:
    """Comment posted when reconsider re-runs triage but the verdict is still fail."""
    missing_lines = _format_missing(verdict.get("missing") or [])
    explanation = verdict.get("explanation") or ""
    noun = "PR" if kind == "pr" else "issue"
    # The trailing HTML marker is used by `seconds_since_last_reconsider_verdict`
    # to enforce a cooldown between repeated `@agent-shin reconsider` triggers.
    return (
        f"⏸️ **Re-evaluated; this {noun} still doesn't meet the rubric.**\n"
        "\n"
        "Agent Shin re-ran triage on the current description but is still "
        "missing:\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "Update the description with the missing pieces and comment "
        "`@agent-shin reconsider` again, or ping a maintainer if you think "
        "I got this wrong.\n"
        "\n"
        "_(I'm an LLM and I'm not infallible.)_\n"
        "\n"
        f"{RECONSIDER_COMMENT_MARKER}"
    )


# ---------------------------------------------------------------------------
# Review gate — "ready for review" label lifecycle

_UNSET = object()


def _combine_missing(
    verdict: dict, greptile_score: int | None, min_score: int
) -> list[str]:
    """Merge the LLM rubric's `missing` list with a Greptile-score shortfall."""
    missing = list(verdict.get("missing") or [])
    if greptile_score is not None and greptile_score < min_score:
        missing.insert(
            0,
            f"Greptile's most recent review scored this PR {greptile_score}/5 "
            f"(below the {min_score}/5 bar)",
        )
    return missing or ["(see explanation below)"]


def _has_marker(
    comments: Iterable[dict], marker: str, *, bot_login: str | None = None
) -> bool:
    """Return True iff the bot itself posted a comment containing ``marker``.

    Filters by author so a contributor who quotes the marker (e.g. via
    GitHub's "Quote reply" feature, which preserves HTML comments in
    raw markdown) is not mistaken for a bot action — that would
    silently suppress notifications or change which "recovered" wording
    is selected. Matches the author-filter pattern used by the sibling
    `_seconds_since_latest_marker_comment` helper.
    """
    expected_login = (
        bot_login
        or os.environ.get("AGENT_SHIN_BOT_LOGIN")
        or AGENT_SHIN_DEFAULT_BOT_LOGIN
    ).lower()
    for comment in comments:
        author = ((comment.get("user") or {}).get("login") or "").lower()
        if author != expected_login:
            continue
        if marker in (comment.get("body") or ""):
            return True
    return False


def format_ready_for_review_comment(
    verdict: dict,
    greptile_score: int | None,
    min_greptile_score: int = DEFAULT_MIN_GREPTILE_SCORE,
) -> str:
    """Posted the first time a PR clears the bar (label added)."""
    score_line = (
        f" Greptile scored it **{greptile_score}/5**."
        if greptile_score is not None
        else ""
    )
    explanation = verdict.get("explanation") or ""
    return (
        "✅ **Triage passed, tagging `ready for review`.**\n"
        "\n"
        "Agent Shin checked this PR against the "
        "[contribution rubric](https://github.com/BerriAI/litellm/blob/main/.github/pull_request_template.md) "
        "and it clears the bar (a linked issue, or a clear problem description "
        f"+ expected vs. actual + QA proof).{score_line}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "A maintainer will take it from here. If a later re-check finds the PR "
        f"has regressed (Greptile drops below {min_greptile_score}/5, "
        "the QA proof is removed, etc.) I'll pull the tag and comment with "
        "what's missing; fix it and the tag comes back automatically.\n"
        f"{READY_MARKER}"
    )


def format_all_clear_comment(verdict: dict, greptile_score: int | None) -> str:
    """Posted when a PR recovers after a regression (label re-added)."""
    score_line = (
        f" Greptile is back to **{greptile_score}/5**."
        if greptile_score is not None
        else ""
    )
    explanation = verdict.get("explanation") or ""
    return (
        "✅ **All clear again, re-adding `ready for review`.**\n"
        "\n"
        "Thanks for addressing the earlier feedback. On re-check this PR meets "
        f"the contribution bar once more.{score_line}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "A maintainer will take another look.\n"
        f"{READY_MARKER}"
    )


def format_regression_comment(
    missing: list[str], explanation: str, grace_days: int
) -> str:
    """Posted when a previously-tagged PR regresses (label removed, PR stays open).

    Discloses the same ``grace_days`` deadline the state machine enforces:
    once that window elapses with the PR still failing, the close path fires.
    Hiding the deadline behind a bare "stays open" would surprise contributors
    with an auto-close they were never warned about.
    """
    window = "24 hours" if grace_days == 1 else f"{grace_days} days"
    return (
        "⚠️ **Removing the `ready for review` tag.**\n"
        "\n"
        "On a re-check this PR no longer meets the contribution bar. What's "
        "missing now:\n"
        "\n"
        f"{_format_missing(missing)}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        f"The PR stays open for ~{window}; address the points above and Agent "
        'Shin will post an "all clear" comment and re-add the tag '
        "automatically. If the points still aren't addressed after that "
        "window, the PR is auto-closed; that's not a rejection, and you can "
        "comment `@agent-shin reconsider` to have it re-evaluated and reopened "
        "once it passes.\n"
        f"{REGRESSED_MARKER}"
    )


def format_within_grace_comment(
    missing: list[str], explanation: str, grace_days: int
) -> str:
    """Posted once while a failing PR is still inside its grace window."""
    window = "24 hours" if grace_days == 1 else f"{grace_days} days"
    return (
        "🚅 Hi, thanks for the PR! This is **Agent Shin**, the automated triage "
        "bot. This PR doesn't quite meet the contribution bar yet:\n"
        "\n"
        f"{_format_missing(missing)}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        f"You have ~{window} from when this PR was opened to add the missing "
        "pieces; just update the description and I'll re-check on the next "
        "sweep. Once it passes I'll tag it `ready for review`. If it does get "
        "auto-closed, that's not a rejection; comment `@agent-shin reconsider` "
        "and I'll re-evaluate and reopen if it now passes.\n"
        f"{WITHIN_GRACE_MARKER}"
    )


def review_gate(
    *,
    repo: str,
    number: int,
    close: bool,
    model: str,
    judge: Any = None,
    greptile_score: Any = _UNSET,
    comments: Any = _UNSET,
    now: dt.datetime | None = None,
    grace_days: int = DEFAULT_GRACE_DAYS,
    min_greptile_score: int = DEFAULT_MIN_GREPTILE_SCORE,
    label: str = READY_FOR_REVIEW_LABEL,
    allowlist: frozenset[str] = ALLOWLIST_LOGINS,
) -> dict:
    """Reconcile the `ready for review` label with a PR's current quality.

    A PR is *passing* when it clears BOTH gates: the LLM rubric (linked issue,
    or problem description + expected/actual + QA proof) AND Greptile's most
    recent confidence score (>= ``min_greptile_score``; absence of a score is
    not held against the PR). The gate then drives a small state machine, using
    the label itself as the persisted state so comments fire only on
    transitions (never on every scheduled run):

      passing, untagged           -> add label + "ready for review" / "all clear"
      passing, tagged             -> noop-passing
      not passing, tagged         -> remove label + regression comment (stays open)
      not passing, untagged, old  -> close + comment (past the grace window)
      not passing, untagged, new  -> one-time "what's missing" notice (within grace)

    ``close`` gates every destructive side effect: with ``close=False`` the
    function returns a ``would-*`` preview and touches nothing, mirroring the
    dry-run contract of :func:`triage`. ``judge``/``greptile_score``/
    ``comments``/``now`` are injectable for tests; in production they are
    resolved from the OpenAI judge, the PR's Greptile comment, the live comment
    list, and the wall clock respectively.
    """
    item = fetch_pr(repo, number)

    title = item.get("title") or ""
    body = item.get("body") or ""
    login = (item.get("user") or {}).get("login") or ""
    association = item.get("author_association") or ""
    state = item.get("state") or ""
    # GitHub label names are case-insensitive; compare lowercased so a repo
    # that already has e.g. "Ready for Review" is recognized as the same
    # label as our READY_FOR_REVIEW_LABEL constant ("ready for review").
    labels_now = {(lbl.get("name") or "").lower() for lbl in (item.get("labels") or [])}
    label_key = label.lower()
    created_raw = item.get("created_at") or ""

    base_result = {
        "kind": "pr",
        "number": number,
        "title": title,
        "author": login,
        "author_association": association,
        "state": state,
        "labeled": label_key in labels_now,
        "review_gate": True,
    }

    if state != "open":
        return {**base_result, "action": "skip-not-open"}

    if allowlist:
        if login.lower() not in allowlist:
            return {**base_result, "action": "skip-not-allowlisted"}
    elif is_internal_contributor(item):
        return {**base_result, "action": "skip-internal-author"}

    # Resolve the comment list once — used for both the Greptile score and the
    # marker-based dedup below.
    if comments is _UNSET:
        comments = list(_iter_paginated_json(f"repos/{repo}/issues/{number}/comments"))

    # --- rubric verdict: linked-issue short-circuit, else the LLM judge -------
    if has_linked_issue(body):
        verdict = {
            "verdict": "pass",
            "linked_issue": True,
            "missing": [],
            "explanation": "Linked-issue regex matched; LLM was not called.",
        }
        rubric_pass = True
    else:
        prompt = build_pr_prompt(title=title, body=body)
        if judge is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return {**base_result, "action": "skip-no-llm-key"}
            base_url = os.environ.get("OPENAI_BASE_URL") or None

            def judge(p: str) -> str:
                return call_llm_judge(
                    p, model=model, api_key=api_key, base_url=base_url
                )

        try:
            verdict = parse_verdict(judge(prompt))
        except Exception as exc:  # noqa: BLE001 - judge errors must never act
            return {**base_result, "action": "skip-llm-error", "error": str(exc)}
        rubric_pass = (verdict.get("verdict") or "").lower() == "pass"

    # --- Greptile score -------------------------------------------------------
    if greptile_score is _UNSET:
        extraction = extract_greptile_score(comments)
        greptile_score = extraction[0] if extraction else None
    greptile_ok = greptile_score is None or greptile_score >= min_greptile_score
    passing = rubric_pass and greptile_ok

    # --- age ------------------------------------------------------------------
    age_days = None
    if created_raw:
        reference = now or dt.datetime.now(dt.timezone.utc)
        age_days = (reference - parse_iso8601(created_raw)).days

    label_present = label_key in labels_now
    explanation = verdict.get("explanation") or ""
    # When the rubric short-circuited to pass (linked-issue regex) but
    # Greptile dragged the PR below the bar, the synthetic verdict's
    # explanation ("LLM was not called") would mislead a contributor reading
    # the regression / close comment. Surface the real reason instead.
    if rubric_pass and not greptile_ok:
        explanation = (
            f"Greptile's most recent review scored this PR "
            f"{greptile_score}/5 (below the {min_greptile_score}/5 bar)."
        )
        verdict = {**verdict, "explanation": explanation}
    base_result = {
        **base_result,
        "verdict": verdict,
        "greptile_score": greptile_score,
        "passing": passing,
        "age_days": age_days,
    }

    if passing:
        if label_present:
            return {**base_result, "action": "noop-passing"}
        recovered = _has_marker(comments, REGRESSED_MARKER)
        comment = (
            format_all_clear_comment(verdict, greptile_score)
            if recovered
            else format_ready_for_review_comment(
                verdict, greptile_score, min_greptile_score
            )
        )
        if not close:
            return {**base_result, "action": "would-label-ready", "comment": comment}
        post_comment(repo, number, comment)
        add_label(repo, number, label)
        return {**base_result, "action": "labeled-ready", "comment": comment}

    missing = _combine_missing(verdict, greptile_score, min_greptile_score)

    if label_present:
        comment = format_regression_comment(missing, explanation, grace_days)
        if not close:
            return {**base_result, "action": "would-remove-label", "comment": comment}
        remove_label(repo, number, label)
        post_comment(repo, number, comment)
        return {**base_result, "action": "label-removed-regressed", "comment": comment}

    # Not passing and not tagged. If the PR was previously tagged and then
    # regressed (we removed the label and posted REGRESSED_MARKER), honor the
    # "PR stays open — fix it and the tag comes back" promise from
    # `format_regression_comment` and skip the close path. Without this guard,
    # any PR older than `grace_days` would be closed on the next evaluation,
    # giving the contributor no realistic window to address the regression.
    #
    # The promise has a deliberate expiration: once `grace_days` have elapsed
    # since the regression notice, fall through to the close path so a PR that
    # was abandoned post-regression doesn't sit open forever.
    if _has_marker(comments, REGRESSED_MARKER):
        reference = now or dt.datetime.now(dt.timezone.utc)
        seconds_since_regression = seconds_since_latest_marker_comment(
            comments, marker=REGRESSED_MARKER, now=reference
        )
        grace_seconds = grace_days * 86400
        if seconds_since_regression is None or seconds_since_regression < grace_seconds:
            return {**base_result, "action": "regressed-already-notified"}

    # Not passing and not tagged: close if past the grace window, else notify once.
    if age_days is not None and age_days >= grace_days:
        comment = format_pr_close_comment({**verdict, "missing": missing})
        if not close:
            return {**base_result, "action": "would-close", "comment": comment}
        post_comment(repo, number, comment)
        close_pr(repo, number)
        return {**base_result, "action": "closed", "comment": comment}

    if _has_marker(comments, WITHIN_GRACE_MARKER):
        return {**base_result, "action": "within-grace-already-notified"}
    comment = format_within_grace_comment(missing, explanation, grace_days)
    if not close:
        return {
            **base_result,
            "action": "would-notify-within-grace",
            "comment": comment,
        }
    post_comment(repo, number, comment)
    return {**base_result, "action": "within-grace-notified", "comment": comment}


def triage(
    *,
    repo: str,
    kind: str,
    number: int,
    close: bool,
    model: str,
    judge: Any = None,
    print_prompt: bool = False,
    reconsider: bool = False,
    allowlist: frozenset[str] = ALLOWLIST_LOGINS,
) -> dict:
    """Triage a single PR or issue. Returns a result dict for logging/tests.

    `judge` is an optional callable `(prompt) -> str` for tests / dry-run with
    a stub. In production, leave it None and the script uses `call_llm_judge`.

    When `reconsider=True`, the closed-state guard is skipped and a
    fail-but-no-comment is replaced with a "still failing" comment + leave
    closed; a pass triggers `reopen_pr`/`reopen_issue` plus a reopen comment.
    Reconsider mode is intended for the `@agent-shin reconsider` comment
    trigger. Like regular triage, `close=False` keeps reconsider in dry-run
    (returns `would-reopen` / `would-reconsider-still-failing` so a local
    operator can preview without write side effects); the workflow only
    passes `--close` when `AGENT_SHIN_ENABLED=true`.

    Reconsider mode adds two extra safety guards on top of the regular
    triage skip-internal-author check:

      1. **Bot-closed guard.** Only reopens if the most recent close was
         performed by the bot identity (default `github-actions[bot]`).
         This stops a contributor from using `@agent-shin reconsider` to
         override a maintainer's close for non-rubric reasons.
      2. **Rate-limit guard.** If the bot has already posted a reconsider
         verdict on this PR/issue within `RECONSIDER_RATE_LIMIT_SECONDS`,
         skip — repeated triggers from the same contributor shouldn't burn
         CI minutes or LLM budget.
    """
    fetcher = {"pr": fetch_pr, "issue": fetch_issue}[kind]
    item = fetcher(repo, number)

    title = item.get("title") or ""
    body = item.get("body") or ""
    login = (item.get("user") or {}).get("login") or ""
    association = item.get("author_association") or ""
    state = item.get("state") or ""

    base_result = {
        "kind": kind,
        "number": number,
        "title": title,
        "author": login,
        "author_association": association,
        "state": state,
        "reconsider": reconsider,
    }

    # Reconsider only makes sense on a closed PR/issue. A "reconsider on an
    # open PR" is a no-op (the regular triage flow already evaluates open
    # PRs); return a clear skip so the workflow can short-circuit.
    if reconsider:
        if state != "closed":
            return {**base_result, "action": "skip-not-closed"}
    else:
        if state != "open":
            return {**base_result, "action": "skip-not-open"}

    if allowlist:
        if login.lower() not in allowlist:
            return {**base_result, "action": "skip-not-allowlisted"}
    elif is_internal_contributor(item):
        return {**base_result, "action": "skip-internal-author"}

    # Reconsider-only guards — these run BEFORE the LLM call so a
    # maintainer-closed PR / rate-limited trigger never spends LLM budget.
    if reconsider:
        if not was_closed_by_agent_shin(repo, number):
            return {**base_result, "action": "skip-not-bot-closed"}
        age = seconds_since_last_reconsider_verdict(repo, number)
        if age is not None and age < RECONSIDER_RATE_LIMIT_SECONDS:
            return {
                **base_result,
                "action": "skip-rate-limited",
                "rate_limit_age_seconds": age,
                "rate_limit_window_seconds": RECONSIDER_RATE_LIMIT_SECONDS,
            }

    if kind == "pr":
        # Short-circuit: if body very clearly links a related issue, just pass.
        if has_linked_issue(body):
            base = {
                **base_result,
                "action": "pass-linked-issue",
                "verdict": {
                    "verdict": "pass",
                    "linked_issue": True,
                    "explanation": "Linked-issue regex matched; LLM was not called.",
                },
            }
            if reconsider:
                # Pass-on-reconsider -> reopen the PR with a friendly comment.
                reopen_body = format_reopen_comment(kind)
                if not close:
                    return {
                        **base,
                        "action": "would-reopen",
                        "comment": reopen_body,
                    }
                post_comment(repo, number, reopen_body)
                reopen_pr(repo, number)
                return {
                    **base,
                    "action": "reopened",
                    "comment": reopen_body,
                }
            return base
        prompt = build_pr_prompt(title=title, body=body)
    else:
        prompt = build_issue_prompt(title=title, body=body)

    if print_prompt:
        return {**base_result, "action": "print-prompt", "prompt": prompt}

    if judge is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # No key configured — never take a destructive action. Report skip.
            return {
                **base_result,
                "action": "skip-no-llm-key",
                "prompt_preview": prompt[:200],
            }
        base_url = os.environ.get("OPENAI_BASE_URL") or None

        def judge(p: str) -> str:
            return call_llm_judge(p, model=model, api_key=api_key, base_url=base_url)

    try:
        raw = judge(prompt)
        verdict = parse_verdict(raw)
    except Exception as exc:  # noqa: BLE001 - judge errors must never close PRs
        return {**base_result, "action": "skip-llm-error", "error": str(exc)}

    decision = (verdict.get("verdict") or "").lower()

    if reconsider:
        # Reconsider: an explicit `pass` -> reopen + post reopen comment;
        # anything else (fail, missing/malformed verdict, typo) -> leave
        # closed + post a "still failing" comment so the contributor can
        # iterate again. Reopen is destructive, so a flaky/empty verdict
        # must not satisfy the gate.
        # In dry-run (`close=False`) we return `would-*` actions instead
        # of touching GitHub state, mirroring the regular triage flow's
        # `would-close`. This lets a local operator preview the outcome
        # of `python triage_with_llm.py --reconsider --pr N` without
        # risking accidental comments or reopens.
        if decision == "pass":
            reopen_body = format_reopen_comment(kind)
            if not close:
                return {
                    **base_result,
                    "action": "would-reopen",
                    "verdict": verdict,
                    "comment": reopen_body,
                }
            post_comment(repo, number, reopen_body)
            if kind == "pr":
                reopen_pr(repo, number)
            else:
                reopen_issue(repo, number)
            return {
                **base_result,
                "action": "reopened",
                "verdict": verdict,
                "comment": reopen_body,
            }
        still_failing = format_reconsider_still_failing_comment(kind, verdict)
        if not close:
            return {
                **base_result,
                "action": "would-reconsider-still-failing",
                "verdict": verdict,
                "comment": still_failing,
            }
        post_comment(repo, number, still_failing)
        return {
            **base_result,
            "action": "reconsider-still-failing",
            "verdict": verdict,
            "comment": still_failing,
        }

    if decision != "fail":
        return {**base_result, "action": "pass-llm", "verdict": verdict}

    # Grace-period flow: on the first low-quality detection, post a warning
    # comment instead of closing immediately. On a subsequent triage run
    # (manual re-trigger, or the daily `close_low_quality_prs.py` cron
    # finding the same PR in its own pass), if `GRACE_PERIOD_SECONDS` has
    # elapsed since the warning AND the PR still fails the rubric, close.
    grace_age = seconds_since_last_grace_warning(repo, number)
    if grace_age is None:
        warning_body = (
            format_grace_warning_pr_comment(verdict)
            if kind == "pr"
            else format_grace_warning_issue_comment(verdict)
        )
        if not close:
            return {
                **base_result,
                "action": "would-warn-grace",
                "verdict": verdict,
                "comment": warning_body,
            }
        post_comment(repo, number, warning_body)
        return {
            **base_result,
            "action": "warned-grace",
            "verdict": verdict,
            "comment": warning_body,
        }
    if grace_age < GRACE_PERIOD_SECONDS:
        return {
            **base_result,
            "action": "skip-in-grace-period",
            "verdict": verdict,
            "grace_age_seconds": grace_age,
            "grace_period_seconds": GRACE_PERIOD_SECONDS,
        }

    # The grace window has elapsed. `--close` still gates the destructive
    # write so a dry-run preview never posts or closes — the workflow only
    # passes `--close` when `AGENT_SHIN_ENABLED=true`, which keeps the bot
    # inert by default.
    if not close:
        return {**base_result, "action": "would-close", "verdict": verdict}

    comment_body = (
        format_pr_close_comment(verdict)
        if kind == "pr"
        else format_issue_close_comment(verdict)
    )
    post_comment(repo, number, comment_body)
    if kind == "pr":
        close_pr(repo, number)
    else:
        close_issue(repo, number)

    return {
        **base_result,
        "action": "closed",
        "verdict": verdict,
        "comment": comment_body,
    }


# ---------------------------------------------------------------------------
# CLI


def render_summary(result: dict) -> str:
    """Render a human-readable summary block (used for stdout + step summary)."""
    lines = ["## Agent Shin verdict", ""]
    lines.append(
        f"- **{result['kind'].upper()} #{result['number']}**: {result.get('title', '')}"
    )
    lines.append(
        f"- **Author**: `{result.get('author', '')}` ({result.get('author_association', '')})"
    )
    lines.append(f"- **State**: {result.get('state', '')}")
    lines.append(f"- **Action**: `{result['action']}`")
    verdict = result.get("verdict")
    if verdict:
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(verdict, indent=2))
        lines.append("```")
    error = result.get("error")
    if error:
        lines.append("")
        lines.append(f"_LLM error: {error}_")
    comment = result.get("comment")
    if comment:
        lines.append("")
        lines.append("### Posted comment:")
        lines.append("")
        lines.append("> " + comment.replace("\n", "\n> "))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Repository (owner/repo).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pr", type=int, help="Pull request number to triage.")
    target.add_argument("--issue", type=int, help="Issue number to triage.")
    parser.add_argument(
        "--close",
        action="store_true",
        help="Actually post comment + close on fail (default: dry run).",
    )
    parser.add_argument(
        "--model",
        # `os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)` would return "" when
        # GitHub Actions exposes an unset repo variable as an empty-string env
        # var, silently bypassing DEFAULT_MODEL and causing every call to fail
        # as `skip-llm-error`. The `or` guard collapses empty -> default.
        default=os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL,
        help=f"OpenAI-compatible model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the prompt that would be sent to the judge and exit.",
    )
    parser.add_argument(
        "--reconsider",
        action="store_true",
        help=(
            "Re-run triage on a CLOSED PR/issue and reopen it on pass. "
            "Used by the `@agent-shin reconsider` comment-trigger workflow. "
            "Only invoke this from a workflow that has already gated on "
            "AGENT_SHIN_ENABLED=true and verified the commenter is the "
            "PR/issue author or an internal collaborator."
        ),
    )
    parser.add_argument(
        "--review-gate",
        action="store_true",
        help=(
            "Reconcile the `ready for review` label for an OPEN PR: tag on "
            "pass, remove the tag + comment on regression, close after the "
            "grace window if it never passed. PR-only."
        ),
    )
    parser.add_argument(
        "--grace-days",
        type=int,
        default=DEFAULT_GRACE_DAYS,
        help=(
            "Review-gate only: hours/24 a failing, un-tagged PR may stay open "
            f"before auto-close (default: {DEFAULT_GRACE_DAYS} = 24h)."
        ),
    )
    parser.add_argument(
        "--min-greptile-score",
        type=int,
        default=DEFAULT_MIN_GREPTILE_SCORE,
        choices=range(1, 6),
        help=(
            "Review-gate only: Greptile score below which a PR counts as not "
            f"passing (default: {DEFAULT_MIN_GREPTILE_SCORE} -> <4/5 regresses)."
        ),
    )
    args = parser.parse_args()

    kind = "pr" if args.pr is not None else "issue"
    number = args.pr if args.pr is not None else args.issue

    if args.review_gate:
        if kind != "pr":
            parser.error("--review-gate applies to pull requests only (use --pr).")
        result = review_gate(
            repo=args.repo,
            number=number,
            close=args.close,
            model=args.model,
            grace_days=args.grace_days,
            min_greptile_score=args.min_greptile_score,
        )
    else:
        result = triage(
            repo=args.repo,
            kind=kind,
            number=number,
            close=args.close,
            model=args.model,
            print_prompt=args.print_prompt,
            reconsider=args.reconsider,
        )

    if result.get("action") == "print-prompt":
        print(result["prompt"])
        return 0

    summary = render_summary(result)
    print(summary)
    write_step_summary(summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
