"""Unit tests for the daily-cron matrix publisher.

The publisher orchestrates Docker, git, npm and pytest — per the PRD's
"Testing Decisions" section, the orchestration itself is too thin to
warrant heavy mocking. These tests cover the small pure helpers that
do warrant test coverage:

- `commit_message_for_matrix`: deterministic commit message containing
  the LiteLLM and Claude Code versions plus `generated_at`, so the docs
  repo's git log shows what produced each push.
- `docker_image_for_tag`: maps a `v*-stable` tag to its `ghcr.io` image.
- `select_files_to_commit`: enforces the "only `compatibility-matrix.json`
  is pushed" guarantee that the GitHub App's broad `contents: write` scope
  doesn't enforce on its own (per PRD: "File-level restriction is enforced
  by script correctness").
"""

from __future__ import annotations

import pytest

from tests.claude_code.publisher import (
    DOCS_TARGET_BASENAME,
    commit_message_for_matrix,
    docker_image_for_tag,
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
