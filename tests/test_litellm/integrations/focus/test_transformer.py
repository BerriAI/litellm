"""Tests for FocusTransformer."""

from __future__ import annotations

import json

import polars as pl

from litellm.integrations.focus.transformer import FocusTransformer


def _make_row(**overrides) -> dict:
    """Return a minimal valid spend-log row, with optional overrides."""
    base = {
        "date": "2026-03-17",
        "spend": 0.05,
        "api_key": "sk-test",
        "api_key_alias": "my-key",
        "model": "claude-sonnet",
        "model_group": "sonnet-group",
        "custom_llm_provider": "anthropic",
        "team_id": "team-1",
        "team_alias": "Engineering",
        "user_id": "user-1",
        "user_email": "test@example.com",
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }
    base.update(overrides)
    return base


def test_service_name_uses_custom_llm_provider_when_present():
    row = _make_row(custom_llm_provider="anthropic", model="claude-sonnet")
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceName"][0] == "anthropic"


def test_service_name_falls_back_to_model_when_provider_blank():
    row = _make_row(custom_llm_provider="", model="claude-sonnet")
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceName"][0] == "claude-sonnet"


def test_service_name_falls_back_to_model_when_provider_null():
    row = _make_row(custom_llm_provider=None, model="claude-sonnet")
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceName"][0] == "claude-sonnet"


def test_service_name_defaults_to_unknown_when_both_blank():
    row = _make_row(custom_llm_provider="", model="")
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceName"][0] == "unknown"


def test_service_name_defaults_to_unknown_when_both_null():
    row = _make_row(custom_llm_provider=None, model=None)
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceName"][0] == "unknown"


def test_service_category_uses_model_group_when_present():
    row = _make_row(model_group="sonnet-group")
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceCategory"][0] == "sonnet-group"


def test_service_category_null_when_model_group_blank():
    row = _make_row(model_group="")
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceCategory"][0] is None


def test_service_category_null_when_model_group_null():
    row = _make_row(model_group=None)
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    assert result["ServiceCategory"][0] is None


def test_tags_include_token_counts():
    row = _make_row(prompt_tokens=100, completion_tokens=50)
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    tags = json.loads(result["Tags"][0])
    assert tags["prompt_tokens"] == "100"
    assert tags["completion_tokens"] == "50"


def test_tags_include_all_metadata_keys():
    row = _make_row()
    frame = pl.DataFrame([row])
    result = FocusTransformer().transform(frame)
    tags = json.loads(result["Tags"][0])
    assert "team_id" in tags
    assert "model" in tags
    assert "prompt_tokens" in tags
    assert "completion_tokens" in tags
