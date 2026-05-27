"""Tests for FocusTransformer — focus on Tags JSON construction for Vantage Token Allocation."""

from __future__ import annotations

import json
from datetime import timedelta

import polars as pl
import pytest

from litellm.integrations.focus.transformer import _TAG_KEYS, FocusTransformer


def _base_row(**overrides):
    row = {
        "date": "2026-05-20",
        "user_id": "user-1",
        "api_key": "sk-abc",
        "model": "gpt-5o",
        "model_group": "gpt-5o",
        "custom_llm_provider": "openai",
        "spend": 1.50,
        "team_id": "team-1",
        "team_alias": "team-one",
        "user_email": "alice@example.com",
        "api_key_alias": "dev-key",
    }
    row.update(overrides)
    return row


def _tags(df: pl.DataFrame, idx: int = 0) -> dict:
    return json.loads(FocusTransformer().transform(df)["Tags"][idx])


def test_tag_keys_include_organization_fields():
    """Both organization_id and organization_alias must be in _TAG_KEYS so Vantage
    Token Allocation can route on org-level data (LIT-2895)."""
    assert "organization_id" in _TAG_KEYS
    assert "organization_alias" in _TAG_KEYS


def test_organization_id_flows_into_tags_when_present():
    row = _base_row(organization_id="org-tempus", organization_alias="Tempus")
    tags = _tags(pl.DataFrame([row]))
    assert tags["organization_id"] == "org-tempus"
    assert tags["organization_alias"] == "Tempus"
    # Existing tag keys still present (no regression)
    assert tags["team_id"] == "team-1"
    assert tags["user_id"] == "user-1"


def test_organization_fields_absent_when_team_has_no_org():
    """When organization_id/alias are NULL (team not in an org), they should
    NOT appear in the Tags JSON (matches existing behavior for nullable keys)."""
    row = _base_row(organization_id=None, organization_alias=None)
    tags = _tags(pl.DataFrame([row]))
    assert "organization_id" not in tags
    assert "organization_alias" not in tags
    # Other keys still flow through
    assert tags["team_id"] == "team-1"


def test_organization_fields_optional_when_columns_missing():
    """If the DB query did not supply organization columns at all (e.g. older
    custom call site), the transformer must not raise; tags simply omit them."""
    row = _base_row()  # no organization_* keys
    tags = _tags(pl.DataFrame([row]))
    assert "organization_id" not in tags
    assert "organization_alias" not in tags
    assert tags["team_id"] == "team-1"


def test_organization_id_only_partial_metadata():
    """A team with organization_id but no resolvable organization_alias still
    flows the id through."""
    row = _base_row(organization_id="org-tempus", organization_alias=None)
    tags = _tags(pl.DataFrame([row]))
    assert tags["organization_id"] == "org-tempus"
    assert "organization_alias" not in tags
