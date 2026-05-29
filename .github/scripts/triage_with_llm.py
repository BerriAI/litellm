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

DEFAULT_MODEL = "gpt-5.4-mini"

INTERNAL_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

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
# regression ("all clear again"). They deliberately do NOT contain
# AGENT_SHIN_AUTO_CLOSE_MARKER, so review-gate chatter on an open PR never
# trips the reconsider provenance check (which keys off the close marker).
READY_MARKER = "<!-- agent-shin:ready -->"
REGRESSED_MARKER = "<!-- agent-shin:regressed -->"
WITHIN_GRACE_MARKER = "<!-- agent-shin:within-grace -->"

# Greptile's GitHub App appears as `greptile-apps[bot]` in REST API comments
# and `greptile-apps` in `gh pr view --json` output. Accept either form. These
# live here (rather than in close_low_quality_prs.py) so both the daily sweep
# and the review gate read the score through one implementation — drift would
# silently let one path close/label a PR the other would spare.
GREPTILE_BOT_LOGINS = frozenset({"greptile-apps", "greptile-apps[bot]"})

# Matches lines like:
#   <h3>Confidence Score: 3/5</h3>
#   **Confidence Score: 4/5**
#   Confidence Score: 5 / 5
SCORE_PATTERN = re.compile(
    r"confidence\s*score\s*[:\-]?\s*(\d+)\s*/\s*5",
    re.IGNORECASE,
)

# Marker phrase Agent Shin always includes in its auto-close comments
# (see `format_pr_close_comment` / `format_issue_close_comment`). The
# provenance check for reconsider matches this marker against a comment
# authored by the same bot login that performed the most recent `closed`
# event, so a contributor cannot reopen a PR/issue that a maintainer
# closed after a prior Agent Shin auto-close. Keep the marker in sync
# with the literal text in those formatter functions.
AGENT_SHIN_AUTO_CLOSE_MARKER = "I'm **Agent Shin**"

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


def gh(*args: str) -> str:
    """Run a `gh` CLI command and return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


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


def fetch_issue_comments(repo: str, number: int) -> list[dict]:
    """Fetch all issue-style comments on a PR/issue (paginated).

    `gh api --paginate` returns one JSON array per page; iterate them and
    flatten. Returns [] on error (the reconsider path treats "no comments
    found" as "no proof Agent Shin closed this", which fails-safe).
    """
    try:
        raw = gh(
            "api",
            "--paginate",
            f"repos/{repo}/issues/{number}/comments?per_page=100",
        )
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
            continue
        if isinstance(parsed, list):
            comments.extend(parsed)
        else:
            comments.append(parsed)
    return comments


def fetch_issue_events(repo: str, number: int) -> list[dict]:
    """Fetch all issue events for a PR/issue (paginated, ascending order).

    Used by the reconsider provenance check to identify the actor of the
    most recent `closed` event. Returns [] on error so the reconsider
    path fails safe (no proof of Agent Shin close -> refuse to reopen).
    """
    try:
        raw = gh(
            "api",
            "--paginate",
            f"repos/{repo}/issues/{number}/events?per_page=100",
        )
    except subprocess.CalledProcessError:
        return []
    events: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            events.extend(parsed)
        else:
            events.append(parsed)
    return events


def was_auto_closed_by_agent_shin(repo: str, number: int) -> bool:
    """Return True iff Agent Shin is responsible for the *current* closure.

    Provenance check for the `@agent-shin reconsider` flow. We require ALL of:

      1. The most recent `closed` event on the PR/issue was performed by
         a bot account (actor login ends with "[bot]"). Agent Shin's
         auto-close workflow uses GH_TOKEN, which posts as
         `github-actions[bot]`. Anchoring on the most recent close — not
         just any historical close — prevents a contributor from
         overriding a *later* maintainer-initiated closure (e.g.
         duplicate, out-of-scope) by polishing the description and
         commenting `@agent-shin reconsider`.
      2. A comment authored by the same bot login that performed the
         close contains the Agent Shin auto-close marker
         (`AGENT_SHIN_AUTO_CLOSE_MARKER`) AND was posted in the current
         open→closed cycle (after the most recent `reopened` event, if
         any, and no later than the most recent `closed` event). This
         anchors the marker to the close that's actually being
         reconsidered: a stale-workflow closure that runs as
         `github-actions[bot]` (because `actions/stale` uses
         `secrets.GITHUB_TOKEN`, the same identity as Agent Shin) does
         NOT post a marker in its own cycle, so the cycle-anchored check
         refuses to override it even though a historical Agent Shin
         marker comment still exists from an earlier cycle.
    """
    events = fetch_issue_events(repo, number)
    last_close_ts = ""
    last_closer = ""
    last_reopen_ts = ""
    for event in events:
        kind = (event.get("event") or "").lower()
        ts = event.get("created_at") or ""
        if kind == "closed":
            last_close_ts = ts
            last_closer = ((event.get("actor") or {}).get("login") or "").lower()
        elif kind == "reopened":
            last_reopen_ts = ts
    if not last_closer or not last_closer.endswith("[bot]"):
        return False
    for comment in fetch_issue_comments(repo, number):
        login = ((comment.get("user") or {}).get("login") or "").lower()
        if login != last_closer:
            continue
        body = comment.get("body") or ""
        if AGENT_SHIN_AUTO_CLOSE_MARKER not in body:
            continue
        comment_ts = comment.get("created_at") or ""
        if last_reopen_ts and comment_ts <= last_reopen_ts:
            continue
        if last_close_ts and comment_ts > last_close_ts:
            continue
        return True
    return False


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
# Greptile score + age helpers (shared with close_low_quality_prs.py)


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

        The PR PASSES triage if it satisfies AT LEAST ONE of:

          (A) It links to a related GitHub issue. Acceptable forms:
              "Fixes #1234", "Closes #1234", "Resolves #1234",
              "Refs https://github.com/BerriAI/litellm/issues/1234". A bare
              issue number without a closing keyword counts only if it's
              clearly the related issue (not a passing mention).

          (B) The PR body contains ALL of:
              - A clear problem description (what bug or missing feature this
                addresses, beyond the title).
              - Expected vs. actual behavior (or, for features, "what's
                possible now vs. with this PR").
              - Visual QA proof: before/after screenshots, a screen recording,
                terminal output, log output, or test output demonstrating the
                fix or feature works end-to-end. Saying "I tested it" is NOT
                proof.

        Bias toward PASS when the PR has structure and context — only FAIL when
        the body is empty, copy-paste filler from the template, or genuinely
        missing both a linked issue AND the core elements of (B).

        Respond with a single JSON object, no prose:

        {{
          "verdict": "pass" | "fail",
          "linked_issue": boolean,
          "has_problem_description": boolean,
          "has_expected_vs_actual": boolean,
          "has_qa_proof": boolean,
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

        For a BUG REPORT the issue PASSES triage when it contains ALL of:
          - A clear reproduction (steps, runnable code snippet, curl command,
            or example config the maintainer can paste into their machine).
          - Screenshot, terminal output, traceback, or log output as proof of
            the bug.
          - Expected vs. actual behavior.

        For a FEATURE REQUEST the issue PASSES triage when it contains ALL of:
          - A clear description of the proposed feature (what should LiteLLM do
            that it does not today).
          - Motivation / use case with a concrete example (config, API call,
            UI flow, or scenario showing what's blocked today).

        Bias toward PASS when the issue has structure and context — only FAIL
        when the body is empty, copy-paste template placeholder text, or a
        one-line "X is broken" with no detail. Asking clarifying questions is
        OK content; mark such issues PASS.

        Respond with a single JSON object, no prose:

        {{
          "verdict": "pass" | "fail",
          "kind": "bug" | "feature" | "other",
          "has_repro": boolean,
          "has_proof": boolean,
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


def format_pr_close_comment(verdict: dict) -> str:
    missing_lines = _format_missing(verdict.get("missing") or [])
    explanation = verdict.get("explanation") or ""
    return (
        f"👋 Hi, thanks for the PR! {AGENT_SHIN_AUTO_CLOSE_MARKER}, the automated triage bot for this repository.\n"
        "\n"
        "This PR is being **auto-closed** because it does not yet meet the bar described in our "
        "[pull-request template](https://github.com/BerriAI/litellm/blob/main/.github/pull_request_template.md). "
        "Specifically, I couldn't find:\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "**This isn't a rejection of the idea.** To bring this PR back:\n"
        "\n"
        "1. Update the PR description to either:\n"
        "   - Link a related GitHub issue (e.g. `Fixes #1234`), OR\n"
        "   - Add a clear **problem description**, **expected vs. actual behavior**, and **visual QA proof** "
        "(before/after screenshots, a short screen recording, or terminal/log output).\n"
        "2. Either:\n"
        "   - **Open a new PR** with the same fixes — recommended path. GitHub does not let external "
        "contributors reopen a PR that was closed by a bot/maintainer, so a fresh PR is the most reliable way "
        "to get back into the review queue.\n"
        "   - **Or** comment `@agent-shin reconsider` on this closed PR after updating the description. "
        "I'll re-run the triage; if it now passes, I'll reopen this PR automatically.\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you — ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, comment "
        "`@agent-shin reconsider` or ping a maintainer — they'll override me.)_"
    )


def format_issue_close_comment(verdict: dict) -> str:
    missing_lines = _format_missing(verdict.get("missing") or [])
    explanation = verdict.get("explanation") or ""
    return (
        f"👋 Hi, thanks for filing this! {AGENT_SHIN_AUTO_CLOSE_MARKER}, the automated triage bot for this repository.\n"
        "\n"
        "This issue is being **auto-closed** because it doesn't yet have enough detail for a maintainer to act on. "
        "Specifically, I couldn't find:\n"
        "\n"
        f"{missing_lines}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "**This isn't a \"won't fix\".** To bring this issue back:\n"
        "\n"
        "1. Edit the issue to add the missing pieces:\n"
        "   - For **bug reports**: a runnable reproduction (code / curl / config), expected vs. actual behavior, "
        "and a screenshot / traceback / log showing the bug.\n"
        "   - For **feature requests**: a concrete description of what should change, plus a use case and example "
        "(config / API call / UI flow).\n"
        "2. Comment `@agent-shin reconsider` on this issue once you've updated it. "
        "I'll re-run triage and reopen the issue if it now meets the bar. "
        "(GitHub doesn't always let the original reporter reopen a bot-closed issue, "
        "so the comment-based reconsider is the reliable path.)\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you — ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, comment "
        "`@agent-shin reconsider` or ping a maintainer — they'll override me.)_"
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
    return (
        f"♻️ **Re-evaluated and reopened.** Thanks for updating the {noun}!\n"
        "\n"
        "Agent Shin re-ran triage on the latest description and it now meets "
        "the bar. A maintainer will take another look soon — please don't "
        f"close this {noun} again unless asked to.\n"
        "\n"
        "_(If a maintainer ends up closing this for non-rubric reasons, that "
        "decision stands; comment `@agent-shin reconsider` again only if you "
        "have substantively new information.)_"
    )


def format_reconsider_still_failing_comment(kind: str, verdict: dict) -> str:
    """Comment posted when reconsider re-runs triage but the verdict is still fail."""
    missing_lines = _format_missing(verdict.get("missing") or [])
    explanation = verdict.get("explanation") or ""
    noun = "PR" if kind == "pr" else "issue"
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
        "_(I'm an LLM and I'm not infallible.)_"
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


def _has_marker(comments: Iterable[dict], marker: str) -> bool:
    return any(marker in (comment.get("body") or "") for comment in comments)


def format_ready_for_review_comment(verdict: dict, greptile_score: int | None) -> str:
    """Posted the first time a PR clears the bar (label added)."""
    score_line = (
        f" Greptile scored it **{greptile_score}/5**."
        if greptile_score is not None
        else ""
    )
    explanation = verdict.get("explanation") or ""
    return (
        "✅ **Triage passed — tagging `ready for review`.**\n"
        "\n"
        "Agent Shin checked this PR against the "
        "[contribution rubric](https://github.com/BerriAI/litellm/blob/main/.github/pull_request_template.md) "
        "and it clears the bar (a linked issue, or a clear problem description "
        f"+ expected vs. actual + QA proof).{score_line}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "A maintainer will take it from here. If a later re-check finds the PR "
        f"has regressed (Greptile drops below {DEFAULT_MIN_GREPTILE_SCORE}/5, "
        "the QA proof is removed, etc.) I'll pull the tag and comment with "
        "what's missing — fix it and the tag comes back automatically.\n"
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
        "✅ **All clear again — re-adding `ready for review`.**\n"
        "\n"
        "Thanks for addressing the earlier feedback. On re-check this PR meets "
        f"the contribution bar once more.{score_line}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        "A maintainer will take another look.\n"
        f"{READY_MARKER}"
    )


def format_regression_comment(missing: list[str], explanation: str) -> str:
    """Posted when a previously-tagged PR regresses (label removed, PR stays open)."""
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
        "The PR stays open — address the points above and Agent Shin will post "
        'an "all clear" comment and re-add the tag automatically.\n'
        f"{REGRESSED_MARKER}"
    )


def format_within_grace_comment(
    missing: list[str], explanation: str, grace_days: int
) -> str:
    """Posted once while a failing PR is still inside its grace window."""
    window = "24 hours" if grace_days == 1 else f"{grace_days} days"
    return (
        "👋 Hi, thanks for the PR! This is **Agent Shin**, the automated triage "
        "bot. This PR doesn't meet the contribution bar yet:\n"
        "\n"
        f"{_format_missing(missing)}\n"
        "\n"
        f"> {explanation}\n"
        "\n"
        f"You have ~{window} from when this PR was opened to add the missing "
        "pieces. Once it passes I'll tag it `ready for review`; otherwise I'll "
        "auto-close it (you can always re-open the conversation with "
        "`@agent-shin reconsider`).\n"
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
    labels_now = {(lbl.get("name") or "") for lbl in (item.get("labels") or [])}
    created_raw = item.get("created_at") or ""

    base_result = {
        "kind": "pr",
        "number": number,
        "title": title,
        "author": login,
        "author_association": association,
        "state": state,
        "labeled": label in labels_now,
        "review_gate": True,
    }

    if state != "open":
        return {**base_result, "action": "skip-not-open"}

    if is_internal_contributor(item):
        return {**base_result, "action": "skip-internal-author"}

    # Resolve the comment list once — used for both the Greptile score and the
    # marker-based dedup below.
    if comments is _UNSET:
        comments = fetch_issue_comments(repo, number)

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

    label_present = label in labels_now
    explanation = verdict.get("explanation") or ""
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
            else format_ready_for_review_comment(verdict, greptile_score)
        )
        if not close:
            return {**base_result, "action": "would-label-ready", "comment": comment}
        post_comment(repo, number, comment)
        add_label(repo, number, label)
        return {**base_result, "action": "labeled-ready", "comment": comment}

    missing = _combine_missing(verdict, greptile_score, min_greptile_score)

    if label_present:
        comment = format_regression_comment(missing, explanation)
        if not close:
            return {**base_result, "action": "would-remove-label", "comment": comment}
        remove_label(repo, number, label)
        post_comment(repo, number, comment)
        return {**base_result, "action": "label-removed-regressed", "comment": comment}

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
) -> dict:
    """Triage a single PR or issue. Returns a result dict for logging/tests.

    `judge` is an optional callable `(prompt) -> str` for tests / dry-run with
    a stub. In production, leave it None and the script uses `call_llm_judge`.

    When `reconsider=True`, the closed-state guard is skipped and a
    fail-but-no-comment is replaced with a "still failing" comment + leave
    closed; a pass triggers `reopen_pr`/`reopen_issue` plus a reopen comment.
    Reconsider mode is intended for the `@agent-shin reconsider` comment
    trigger.

    `close` still controls whether destructive side effects fire. When
    `close=False` and `reconsider=True`, the function previews the
    decision (`would-reopen`, `would-leave-closed-still-failing`, or
    `skip-not-bot-closed`) without posting comments or reopening — this
    mirrors the regular `would-close` dry-run behavior so the reconsider
    workflow can be exercised safely with `AGENT_SHIN_ENABLED != "true"`.

    Provenance: when `reconsider=True`, we additionally check that the
    PR/issue was actually auto-closed by Agent Shin (via the bot-authored
    auto-close marker comment). If not, we refuse to reopen so a
    maintainer-closed PR cannot be silently overridden by the author
    polishing the description and commenting `@agent-shin reconsider`.
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

    if is_internal_contributor(item):
        return {**base_result, "action": "skip-internal-author"}

    # Provenance gate for reconsider: only reopen items Agent Shin auto-closed.
    # Done up-front so a maintainer-closed PR short-circuits before we burn
    # LLM tokens or touch comments. The check requires a bot-authored
    # comment containing the auto-close marker (see
    # `was_auto_closed_by_agent_shin`), so a contributor cannot spoof it by
    # quoting the close template themselves.
    if reconsider and not was_auto_closed_by_agent_shin(repo, number):
        return {**base_result, "action": "skip-not-bot-closed"}

    if kind == "pr":
        prompt = build_pr_prompt(title=title, body=body)
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
                reopen_body = format_reopen_comment(kind)
                if not close:
                    return {
                        **base,
                        "action": "would-reopen",
                        "comment": reopen_body,
                    }
                # Reopen before posting so a failed reopen call doesn't
                # leave a misleading "we reopened it" comment on a still-
                # closed PR.
                reopen_pr(repo, number)
                post_comment(repo, number, reopen_body)
                return {
                    **base,
                    "action": "reopened",
                    "comment": reopen_body,
                }
            return base
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
        # Reconsider: pass -> reopen + post reopen comment;
        # fail -> leave closed + post a "still failing" comment so the
        # contributor can iterate again. When `close=False` we preview
        # the action instead of actually posting/reopening.
        # Only an explicit "pass" verdict triggers a reopen — any other
        # value (including missing/malformed verdicts) is fail-safe and
        # leaves the PR/issue closed.
        if decision == "pass":
            reopen_body = format_reopen_comment(kind)
            if not close:
                return {
                    **base_result,
                    "action": "would-reopen",
                    "verdict": verdict,
                    "comment": reopen_body,
                }
            # Reopen before posting so a failed reopen call doesn't leave a
            # misleading "reopened" comment on a still-closed item.
            if kind == "pr":
                reopen_pr(repo, number)
            else:
                reopen_issue(repo, number)
            post_comment(repo, number, reopen_body)
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
                "action": "would-leave-closed-still-failing",
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
    if result.get("review_gate"):
        score = result.get("greptile_score")
        lines.append(
            f"- **Greptile**: {score}/5"
            if score is not None
            else "- **Greptile**: (no score yet)"
        )
        lines.append(f"- **`ready for review` label present**: {result.get('labeled')}")
        if result.get("age_days") is not None:
            lines.append(f"- **Age**: {result['age_days']}d")
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
