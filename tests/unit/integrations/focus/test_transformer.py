"""Tests for FocusTransformer metadata in Tags."""

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


def test_should_include_project_tags_only_when_present_and_keep_team_subaccounts():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 3,
            "spend": [1.25] * 3,
            "api_requests": [1] * 3,
            "api_key": ["scoped-key", "unnamed-project-key", "unscoped-key"],
            "api_key_alias": ["scoped", "unnamed-project", "unscoped"],
            "model": ["gpt-4o"] * 3,
            "model_group": ["gpt-4o"] * 3,
            "custom_llm_provider": ["openai"] * 3,
            "team_id": ["team-1"] * 3,
            "team_alias": ["Platform"] * 3,
            "project_id": ["project-1", "project-2", None],
            "project_alias": ["Production", None, None],
        }
    )

    normalized = FocusTransformer().transform(frame)

    tags = [json.loads(value) for value in normalized["Tags"]]
    assert [{key: value for key, value in row.items() if key.startswith("project_")} for row in tags] == [
        {"project_id": "project-1", "project_alias": "Production"},
        {"project_id": "project-2"},
        {},
    ]
    assert normalized.select("SubAccountId", "SubAccountName").rows() == [("team-1", "Platform")] * 3
