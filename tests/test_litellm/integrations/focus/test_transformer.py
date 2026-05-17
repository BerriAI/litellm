"""Tests for FocusTransformer normalization."""

from __future__ import annotations

import polars as pl

from litellm.integrations.focus.transformer import FocusTransformer


def _usage_frame(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "date": "2026-05-17",
        "user_id": "user-1",
        "api_key": "virtual-key-id",
        "api_key_alias": "test-key",
        "model": "gpt-4.1",
        "model_group": "gpt-4.1",
        "custom_llm_provider": "openai",
        "spend": 0.1,
        "team_id": "team-1",
        "team_alias": "Test Team",
        "user_email": "user@example.com",
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_should_use_model_group_as_service_name_when_present():
    result = FocusTransformer().transform(
        _usage_frame([{"model": "gpt-4.1", "model_group": "production-gpt"}])
    )

    assert result["ServiceName"].to_list() == ["production-gpt"]


def test_should_fallback_service_name_when_model_group_is_blank():
    result = FocusTransformer().transform(
        _usage_frame(
            [
                {"model": "mcp-vector-store", "model_group": ""},
                {"model": "", "model_group": "  ", "custom_llm_provider": "proxy"},
                {"model": None, "model_group": None, "custom_llm_provider": None},
            ]
        )
    )

    assert result["ServiceName"].to_list() == [
        "mcp-vector-store",
        "proxy",
        "litellm-proxy",
    ]
    assert all(service_name.strip() for service_name in result["ServiceName"])
