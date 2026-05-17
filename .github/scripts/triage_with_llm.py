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
import json
import os
import re
import subprocess
import sys
import textwrap
from typing import Any

DEFAULT_MODEL = "gpt-5.4-mini"

INTERNAL_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

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
    template = textwrap.dedent(
        """
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
        """
    ).strip()
    return template.format(title=title, cleaned_body=cleaned_body)


def build_issue_prompt(*, title: str, body: str) -> str:
    cleaned_body = strip_html_comments(body or "").strip() or "(empty)"
    # Dedent the static template *before* interpolating dynamic fields so that
    # multi-line bodies (whose 2nd+ lines start at column 0) don't defeat the
    # common-indent computation in textwrap.dedent.
    template = textwrap.dedent(
        """
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
        """
    ).strip()
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
        "👋 Hi, thanks for the PR! I'm **Agent Shin**, the automated triage bot for this repository.\n"
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
        "2. **Reopen** the PR (or open a fresh one) — I'll re-evaluate automatically.\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you — ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, reopen and ping a maintainer — "
        "they'll override me.)_"
    )


def format_issue_close_comment(verdict: dict) -> str:
    missing_lines = _format_missing(verdict.get("missing") or [])
    explanation = verdict.get("explanation") or ""
    return (
        "👋 Hi, thanks for filing this! I'm **Agent Shin**, the automated triage bot for this repository.\n"
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
        "2. **Reopen** the issue — I'll re-evaluate automatically.\n"
        "\n"
        "Internal BerriAI contributors: this rubric doesn't apply to you — ping a maintainer.\n"
        "\n"
        "_(I'm an LLM, so I'm not infallible. If you think I got this wrong, reopen and ping a maintainer — "
        "they'll override me.)_"
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


def triage(
    *,
    repo: str,
    kind: str,
    number: int,
    close: bool,
    model: str,
    judge: Any = None,
    print_prompt: bool = False,
) -> dict:
    """Triage a single PR or issue. Returns a result dict for logging/tests.

    `judge` is an optional callable `(prompt) -> str` for tests / dry-run with
    a stub. In production, leave it None and the script uses `call_llm_judge`.
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
    }

    if state != "open":
        return {**base_result, "action": "skip-not-open"}

    if is_internal_contributor(item):
        return {**base_result, "action": "skip-internal-author"}

    if kind == "pr":
        prompt = build_pr_prompt(title=title, body=body)
        # Short-circuit: if body very clearly links a related issue, just pass.
        if has_linked_issue(body):
            return {
                **base_result,
                "action": "pass-linked-issue",
                "verdict": {
                    "verdict": "pass",
                    "linked_issue": True,
                    "explanation": "Linked-issue regex matched; LLM was not called.",
                },
            }
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
    args = parser.parse_args()

    kind = "pr" if args.pr is not None else "issue"
    number = args.pr if args.pr is not None else args.issue

    result = triage(
        repo=args.repo,
        kind=kind,
        number=number,
        close=args.close,
        model=args.model,
        print_prompt=args.print_prompt,
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
