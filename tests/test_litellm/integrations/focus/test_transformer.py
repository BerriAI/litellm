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
