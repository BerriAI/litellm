"""Schema definitions for Focus export data."""

from __future__ import annotations

import polars as pl


FOCUS_NORMALIZED_SCHEMA = pl.Schema(
    {
        "usage_date": pl.Datetime(time_unit="us"),
        "team_id": pl.String,
        "team_alias": pl.String,
        "user_id": pl.String,
        "user_email": pl.String,
        "api_key_alias": pl.String,
        "model": pl.String,
        "model_group": pl.String,
        "custom_llm_provider": pl.String,
        "prompt_tokens": pl.Int64,
        "completion_tokens": pl.Int64,
        "total_tokens": pl.Int64,
        "spend": pl.Float64,
        "cache_creation_input_tokens": pl.Int64,
        "cache_read_input_tokens": pl.Int64,
        "api_requests": pl.Int64,
        "successful_requests": pl.Int64,
        "failed_requests": pl.Int64,
        "created_at": pl.Datetime(time_unit="us"),
        "updated_at": pl.Datetime(time_unit="us"),
    }
)

__all__ = ["FOCUS_NORMALIZED_SCHEMA"]
