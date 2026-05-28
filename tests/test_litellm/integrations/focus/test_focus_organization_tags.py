"""Regression tests for LIT-2895 - FOCUS export must include organization_id
and organization_alias in the Tags JSON so Vantage Token Allocation can route
on org-level rollups.

Tests the two-file fix:
  - litellm/integrations/focus/transformer.py - _TAG_KEYS exposes the new keys
    and the per-row Tags JSON propagates them when present.
  - litellm/integrations/focus/database.py - the SQL SELECT picks up
    organization_id (from LiteLLM_TeamTable) and organization_alias (via
    LEFT JOIN LiteLLM_OrganizationTable).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import json
import polars as pl
import pytest

from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.transformer import _TAG_KEYS, FocusTransformer


def _setup_db(monkeypatch, query_return):
    query_mock = AsyncMock(return_value=query_return)
    mock_client = SimpleNamespace(db=SimpleNamespace(query_raw=query_mock))
    db = FocusLiteLLMDatabase()
    monkeypatch.setattr(db, "_ensure_prisma_client", lambda: mock_client)
    return db, query_mock


def test_tag_keys_include_organization_fields():
    assert "organization_id" in _TAG_KEYS
    assert "organization_alias" in _TAG_KEYS


def test_tag_keys_organization_keys_precede_team_keys():
    keys = list(_TAG_KEYS)
    assert keys.index("organization_id") < keys.index("team_id")
    assert keys.index("organization_alias") < keys.index("team_alias")


def _make_frame(**overrides):
    base = {
        "date": ["2026-01-01"],
        "user_id": ["u1"],
        "api_key": ["sk-abc"],
        "api_key_alias": ["dev-key"],
        "model": ["gpt-5-mini"],
        "model_group": ["openai-pool"],
        "custom_llm_provider": ["openai"],
        "prompt_tokens": [10],
        "completion_tokens": [5],
        "spend": [0.001],
        "api_requests": [1],
        "successful_requests": [1],
        "failed_requests": [0],
        "cache_creation_input_tokens": [0],
        "cache_read_input_tokens": [0],
        "team_id": ["t1"],
        "team_alias": ["finance-team"],
        "user_email": ["alice@example.com"],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_transform_includes_organization_fields_in_tags_when_present():
    frame = _make_frame(
        organization_id=["org-1"],
        organization_alias=["finance-org"],
    )
    out = FocusTransformer().transform(frame)
    tags_row = json.loads(out["Tags"][0])
    assert tags_row["organization_id"] == "org-1"
    assert tags_row["organization_alias"] == "finance-org"
    assert tags_row["team_id"] == "t1"
    assert tags_row["api_key_alias"] == "dev-key"


def test_transform_omits_organization_fields_when_team_has_no_org():
    frame = _make_frame(
        organization_id=[None],
        organization_alias=[None],
    )
    out = FocusTransformer().transform(frame)
    tags_row = json.loads(out["Tags"][0])
    assert "organization_id" not in tags_row
    assert "organization_alias" not in tags_row
    assert tags_row["team_id"] == "t1"


def test_transform_works_when_organization_columns_absent_entirely():
    out = FocusTransformer().transform(_make_frame())
    tags_row = json.loads(out["Tags"][0])
    assert "organization_id" not in tags_row
    assert "organization_alias" not in tags_row
    assert tags_row["team_id"] == "t1"


@pytest.mark.asyncio
async def test_database_query_selects_organization_columns(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    await db.get_usage_data()
    query_text, *_ = query_mock.await_args.args
    assert "tt.organization_id as organization_id" in query_text
    assert "ot.organization_alias as organization_alias" in query_text


@pytest.mark.asyncio
async def test_database_query_joins_organization_table(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    await db.get_usage_data()
    query_text, *_ = query_mock.await_args.args
    join_clause = (
        'LEFT JOIN "LiteLLM_OrganizationTable" ot '
        'ON tt.organization_id = ot.organization_id'
    )
    assert join_clause in query_text


@pytest.mark.asyncio
async def test_database_query_emits_organization_join_only_once(monkeypatch):
    db, query_mock = _setup_db(monkeypatch, [])
    await db.get_usage_data()
    query_text, *_ = query_mock.await_args.args
    assert query_text.count('LEFT JOIN "LiteLLM_OrganizationTable"') == 1
