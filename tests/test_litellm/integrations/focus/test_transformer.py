"""Tests for FocusTransformer organization metadata in Tags."""

from __future__ import annotations

import json
from datetime import date

import polars as pl

from litellm.integrations.focus.transformer import FocusTransformer


def test_should_include_organization_fields_in_tags():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "spend": [1.25],
            "api_requests": [1],
            "api_key": ["hashed-key"],
            "api_key_alias": ["prod-key"],
            "model": ["gpt-4o"],
            "model_group": ["gpt-4o"],
            "custom_llm_provider": ["openai"],
            "team_id": ["team-1"],
            "team_alias": ["Platform"],
            "organization_id": ["org-123"],
            "organization_alias": ["Acme Corp"],
            "user_id": ["user-1"],
            "user_email": ["user@example.com"],
        }
    )

    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    assert tags["organization_id"] == "org-123"
    assert tags["organization_alias"] == "Acme Corp"
    assert tags["team_id"] == "team-1"


def test_should_include_request_tags_in_tags():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "spend": [1.25],
            "api_requests": [1],
            "api_key": ["hashed-key"],
            "api_key_alias": ["prod-key"],
            "model": ["gpt-4o"],
            "model_group": ["gpt-4o"],
            "custom_llm_provider": ["openai"],
            "team_id": ["team-1"],
            "team_alias": ["Platform"],
            "request_tags": [["prod", "checkout"]],
        }
    )

    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    assert json.loads(tags["request_tags"]) == ["prod", "checkout"]
    assert tags["team_id"] == "team-1"
    assert "request_tags_truncated" not in tags


def test_should_cap_unbounded_request_tags():
    """Caller-supplied tags are unbounded; the export must cap how many and how
    long they are so one row cannot exceed a destination's per-row size limit,
    and must record the true count via request_tags_truncated."""
    long_tag = "x" * 500
    many_tags = [f"tag-{i}" for i in range(70)]
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "spend": [1.0],
            "api_requests": [1],
            "api_key": ["hashed-key"],
            "api_key_alias": ["prod-key"],
            "model": ["gpt-4o"],
            "model_group": ["gpt-4o"],
            "custom_llm_provider": ["openai"],
            "team_id": ["team-1"],
            "team_alias": ["Platform"],
            "request_tags": [[long_tag, *many_tags]],
        }
    )

    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    emitted = json.loads(tags["request_tags"])
    assert len(emitted) == 64  # _MAX_REQUEST_TAGS
    assert all(len(t) <= 128 for t in emitted)  # _MAX_TAG_LENGTH
    assert tags["request_tags_truncated"] == "71"  # 1 long + 70


def test_should_not_mark_truncated_at_exactly_the_cap():
    """Exactly _MAX_REQUEST_TAGS tags is not truncation: all are kept and no
    marker is emitted (guards the `>` vs `>=` boundary)."""
    exactly_cap = [f"tag-{i}" for i in range(64)]
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "spend": [1.0],
            "api_requests": [1],
            "api_key": ["hashed-key"],
            "api_key_alias": ["prod-key"],
            "model": ["gpt-4o"],
            "model_group": ["gpt-4o"],
            "custom_llm_provider": ["openai"],
            "team_id": ["team-1"],
            "team_alias": ["Platform"],
            "request_tags": [exactly_cap],
        }
    )

    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    assert len(json.loads(tags["request_tags"])) == 64
    assert "request_tags_truncated" not in tags


def test_should_omit_request_tags_when_empty():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "spend": [1.25],
            "api_requests": [1],
            "api_key": ["hashed-key"],
            "api_key_alias": ["prod-key"],
            "model": ["gpt-4o"],
            "model_group": ["gpt-4o"],
            "custom_llm_provider": ["openai"],
            "team_id": ["team-1"],
            "team_alias": ["Platform"],
        }
    ).with_columns(
        pl.Series("request_tags", [None], dtype=pl.List(pl.String)),
    )

    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    assert "request_tags" not in tags
    assert tags["team_id"] == "team-1"


def test_should_omit_missing_organization_fields_from_tags():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "spend": [0.5],
            "api_requests": [1],
            "api_key": ["hashed-key"],
            "api_key_alias": ["prod-key"],
            "model": ["gpt-4o-mini"],
            "model_group": ["gpt-4o-mini"],
            "custom_llm_provider": ["openai"],
            "team_id": ["team-1"],
            "team_alias": ["Platform"],
        }
    )

    normalized = FocusTransformer().transform(frame)

    tags = json.loads(normalized["Tags"][0])
    assert "organization_id" not in tags
    assert "organization_alias" not in tags
    assert tags["team_id"] == "team-1"
