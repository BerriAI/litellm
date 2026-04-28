"""Unit tests for the daily-cron matrix publisher.

The publisher orchestrates Docker, git, npm, gh and pytest — per the
PRD's "Testing Decisions" section, the orchestration itself is too thin
to warrant heavy mocking. These tests cover the small pure helpers that
do warrant test coverage:

- `commit_message_for_matrix`: deterministic commit message containing
  the LiteLLM and Claude Code versions plus `generated_at`, so the docs
  repo's git log shows what produced each push.
- `docker_image_for_tag`: maps a `v*-stable` tag to its `ghcr.io` image.
- `select_files_to_commit`: enforces the "only `compatibility-matrix.json`
  is pushed" guarantee that the GitHub App's broad `contents: write` scope
  doesn't enforce on its own (per PRD: "File-level restriction is enforced
  by script correctness").
- `pr_branch_name`: deterministic head-branch name for the docs PR.
  Two cron runs on the same UTC day with the same resolved versions
  must collide on this branch so the second run updates the existing
  PR rather than spawning a new one.
- `pr_title_for_matrix` / `pr_body_for_matrix`: the strings the publisher
  hands to `gh pr create`. Tested for content (not formatting trivia)
  to keep the tests resilient to copy edits.
"""

from __future__ import annotations

import pytest

from tests.claude_code.publisher import (
    DOCS_TARGET_BASENAME,
    PR_BRANCH_PREFIX,
    commit_message_for_matrix,
    docker_image_for_tag,
    pr_body_for_matrix,
    pr_branch_name,
    pr_title_for_matrix,
    select_files_to_commit,
)


def test_commit_message_includes_versions_and_timestamp():
    matrix = {
        "schema_version": "1",
        "generated_at": "2026-04-25T06:00:00Z",
        "litellm_version": "v1.83.0-stable",
        "claude_code_version": "2.1.120",
        "providers": ["anthropic"],
        "features": [],
    }
    message = commit_message_for_matrix(matrix)
    # Headline is short and identifies the artifact.
    headline, _, body = message.partition("\n")
    assert "compatibility matrix" in headline.lower()
    # Body must surface the three pieces of provenance the docs banner shows.
    assert "v1.83.0-stable" in body
    assert "2.1.120" in body
    assert "2026-04-25T06:00:00Z" in body


def test_commit_message_is_deterministic():
    """Same matrix in → same message out (no clock, no randomness)."""
    matrix = {
        "litellm_version": "v1.83.0-stable",
        "claude_code_version": "2.1.120",
        "generated_at": "2026-04-25T06:00:00Z",
    }
    assert commit_message_for_matrix(matrix) == commit_message_for_matrix(matrix)


def test_docker_image_for_tag_targets_berriai_ghcr():
    assert (
        docker_image_for_tag("v1.83.0-stable")
        == "ghcr.io/berriai/litellm:v1.83.0-stable"
    )


def test_docker_image_for_tag_rejects_empty_tag():
    with pytest.raises(ValueError, match="tag must be a non-empty string"):
        docker_image_for_tag("")


def test_select_files_to_commit_drops_anything_other_than_matrix():
    """The script's safety net: even if pytest leaves stray artifacts in
    the docs-repo checkout, only `compatibility-matrix.json` ever ships.

    Acceptance criterion: "the script does not write any other files".
    """
    staged = [
        "static/data/compatibility-matrix.json",
        "static/data/notes.txt",
        ".github/workflows/secret.yml",
        "compat-results.json",
    ]
    assert select_files_to_commit(staged, DOCS_TARGET_BASENAME) == [
        "static/data/compatibility-matrix.json"
    ]


def test_select_files_to_commit_returns_empty_when_nothing_matches():
    assert select_files_to_commit(["other.json"], DOCS_TARGET_BASENAME) == []


def test_select_files_to_commit_basename_match_not_substring():
    """`compatibility-matrix.json.bak` must not be treated as the allowed file."""
    assert (
        select_files_to_commit(
            ["static/data/compatibility-matrix.json.bak"], DOCS_TARGET_BASENAME
        )
        == []
    )


# ---------------------------------------------------------------------------
# PR-creation helpers
# ---------------------------------------------------------------------------


def test_pr_branch_name_is_deterministic_per_inputs():
    """Same (litellm, claude, date) -> same branch -> same PR.

    This is the idempotency contract for re-runs on the same UTC day:
    the second push lands on the existing branch, `gh pr create` no-ops
    because the PR already exists, and the docs-repo PR list stays
    clean.
    """
    a = pr_branch_name(
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        date_utc="2026-04-25",
    )
    b = pr_branch_name(
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        date_utc="2026-04-25",
    )
    assert a == b
    assert a.startswith(f"{PR_BRANCH_PREFIX}/")
    # All three components must appear so a maintainer can read the
    # provenance off the branch name without opening the PR body.
    assert "v1.83.0-stable" in a
    assert "2.1.120" in a
    assert "2026-04-25" in a


def test_pr_branch_name_changes_when_any_component_changes():
    """A bump in any component must produce a fresh branch (and thus PR)."""
    base = pr_branch_name(
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        date_utc="2026-04-25",
    )
    new_litellm = pr_branch_name(
        litellm_version="v1.84.0-stable",
        claude_code_version="2.1.120",
        date_utc="2026-04-25",
    )
    new_claude = pr_branch_name(
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.121",
        date_utc="2026-04-25",
    )
    new_date = pr_branch_name(
        litellm_version="v1.83.0-stable",
        claude_code_version="2.1.120",
        date_utc="2026-04-26",
    )
    assert len({base, new_litellm, new_claude, new_date}) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "litellm_version": "",
            "claude_code_version": "2.1.120",
            "date_utc": "2026-04-25",
        },
        {
            "litellm_version": "v1.83.0-stable",
            "claude_code_version": "",
            "date_utc": "2026-04-25",
        },
        {
            "litellm_version": "v1.83.0-stable",
            "claude_code_version": "2.1.120",
            "date_utc": "",
        },
    ],
)
def test_pr_branch_name_rejects_empty_components(kwargs):
    """Empty inputs would silently collapse two distinct PRs onto one branch."""
    with pytest.raises(ValueError):
        pr_branch_name(**kwargs)


def test_pr_title_includes_versions_inline():
    """The title is what shows up in notification emails / the PR list,
    so the LiteLLM and Claude Code versions must be inline."""
    matrix = {
        "litellm_version": "v1.83.0-stable",
        "claude_code_version": "2.1.120",
        "generated_at": "2026-04-25T06:00:00Z",
    }
    title = pr_title_for_matrix(matrix)
    assert "v1.83.0-stable" in title
    assert "2.1.120" in title
    # Single-line — newlines in a PR title would render as a literal `\n`.
    assert "\n" not in title


def test_pr_body_renders_per_feature_status_table():
    """Reviewers triage the PR off the body without opening the diff,
    so each feature must list its per-provider status, in manifest order."""
    matrix = {
        "litellm_version": "v1.83.0-stable",
        "claude_code_version": "2.1.120",
        "generated_at": "2026-04-25T06:00:00Z",
        "providers": ["anthropic", "bedrock_invoke"],
        "features": [
            {
                "id": "basic_messaging_non_streaming",
                "name": "Basic messaging (non-streaming)",
                "providers": {
                    "anthropic": {"status": "pass"},
                    "bedrock_invoke": {"status": "fail", "error": "boom"},
                },
            },
            {
                "id": "tool_use",
                "name": "Tool use",
                "providers": {
                    "anthropic": {"status": "pass"},
                    "bedrock_invoke": {
                        "status": "not_applicable",
                        "reason": "tier mismatch",
                    },
                },
            },
        ],
    }
    body = pr_body_for_matrix(matrix)
    # Provenance header.
    assert "v1.83.0-stable" in body
    assert "2.1.120" in body
    assert "2026-04-25T06:00:00Z" in body
    # Per-feature lines surface the names and statuses inline.
    assert "Basic messaging (non-streaming)" in body
    assert "anthropic=pass" in body
    assert "bedrock_invoke=fail" in body
    assert "Tool use" in body
    assert "bedrock_invoke=not_applicable" in body
    # Provider order in each row follows the `providers` list, not dict
    # iteration of `.providers` — so a renamed provider in the matrix
    # without a manifest update would visibly be absent rather than
    # silently drift.
    feature_line = next(
        line for line in body.splitlines() if line.startswith("- **Basic")
    )
    assert feature_line.index("anthropic=") < feature_line.index("bedrock_invoke=")


def test_pr_body_handles_missing_providers_in_a_feature():
    """A feature that didn't run on a manifest provider must still appear
    in that column — as `not_tested` — so the table stays rectangular."""
    matrix = {
        "litellm_version": "v1.83.0-stable",
        "claude_code_version": "2.1.120",
        "generated_at": "2026-04-25T06:00:00Z",
        "providers": ["anthropic", "bedrock_invoke"],
        "features": [
            {
                "id": "vision",
                "name": "Vision",
                "providers": {"anthropic": {"status": "pass"}},
            }
        ],
    }
    body = pr_body_for_matrix(matrix)
    assert "anthropic=pass" in body
    assert "bedrock_invoke=not_tested" in body
