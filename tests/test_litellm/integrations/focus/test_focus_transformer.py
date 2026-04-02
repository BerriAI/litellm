"""Tests for FocusTransformer (default) and FocusSkuTransformer (SKU breakdown)."""

from __future__ import annotations

import polars as pl
import pytest

from litellm.integrations.focus.transformer import (
    FocusSkuTransformer,
    FocusTransformer,
    _explode_by_sku,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(**overrides) -> dict:
    """Return a minimal DB row dict with sensible defaults."""
    base = {
        "id": 1,
        "date": "2024-03-15",
        "user_id": "user_abc",
        "api_key": "sk-hashed-key",
        "api_key_alias": "my-key",
        "model": "gpt-4o",
        "model_group": "openai",
        "custom_llm_provider": "openai",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "spend": 0.003,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "team_id": "team_xyz",
        "team_alias": "my-team",
        "user_email": "user@example.com",
    }
    base.update(overrides)
    return base


def _frame(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# FocusTransformer (default / backwards-compatible) tests
# ---------------------------------------------------------------------------


def test_default_transformer_empty_frame():
    result = FocusTransformer().transform(pl.DataFrame())
    assert result.is_empty()


def test_default_transformer_one_row_per_input_row():
    """Default transformer must NOT explode rows — one input → one output."""
    frame = _frame(_make_row(prompt_tokens=100, completion_tokens=50))
    result = FocusTransformer().transform(frame)
    assert result.height == 1


def test_default_transformer_consumed_unit_is_requests():
    frame = _frame(_make_row())
    result = FocusTransformer().transform(frame)
    assert result["ConsumedUnit"][0] == "Requests"


def test_default_transformer_service_subcategory_is_generative_ai():
    frame = _frame(_make_row())
    result = FocusTransformer().transform(frame)
    assert result["ServiceSubcategory"][0] == "Generative AI"


def test_default_transformer_resource_id_is_model():
    frame = _frame(_make_row(model="gpt-4o"))
    result = FocusTransformer().transform(frame)
    assert result["ResourceId"][0] == "gpt-4o"


def test_default_transformer_billed_cost_is_total_spend():
    frame = _frame(_make_row(spend=0.05))
    result = FocusTransformer().transform(frame)
    assert float(result["BilledCost"][0]) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# _explode_by_sku unit tests
# ---------------------------------------------------------------------------


def test_explode_empty_frame_returns_empty_with_sku_columns():
    empty = pl.DataFrame(
        {
            "prompt_tokens": pl.Series([], dtype=pl.Int64),
            "completion_tokens": pl.Series([], dtype=pl.Int64),
            "spend": pl.Series([], dtype=pl.Float64),
        }
    )
    result = _explode_by_sku(empty)
    assert result.is_empty()
    assert "_sku_type" in result.columns
    assert "_sku_quantity" in result.columns


def test_explode_single_row_prompt_and_completion():
    frame = _frame(_make_row(prompt_tokens=10, completion_tokens=5))
    result = _explode_by_sku(frame)

    sku_types = set(result["_sku_type"].to_list())
    assert sku_types == {"prompt_token", "completion_token"}
    assert result.height == 2


def test_explode_zero_tokens_excluded():
    frame = _frame(_make_row(prompt_tokens=20, completion_tokens=0))
    result = _explode_by_sku(frame)

    assert result.height == 1
    assert result["_sku_type"][0] == "prompt_token"


def test_explode_all_four_sku_types():
    frame = _frame(
        _make_row(
            prompt_tokens=10,
            completion_tokens=5,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=2,
        )
    )
    result = _explode_by_sku(frame)

    sku_types = set(result["_sku_type"].to_list())
    assert sku_types == {
        "prompt_token",
        "completion_token",
        "cache_read_token",
        "cache_creation_token",
    }
    assert result.height == 4


def test_explode_preserves_sku_quantity():
    frame = _frame(_make_row(prompt_tokens=42, completion_tokens=7))
    result = _explode_by_sku(frame)

    qty_by_sku = {
        row["_sku_type"]: row["_sku_quantity"]
        for row in result.iter_rows(named=True)
    }
    assert qty_by_sku["prompt_token"] == 42.0
    assert qty_by_sku["completion_token"] == 7.0


def test_explode_null_token_treated_as_zero():
    frame = pl.DataFrame(
        [
            {
                "prompt_tokens": 10,
                "completion_tokens": None,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "spend": 0.001,
                "date": "2024-01-01",
                "model": "gpt-4",
                "model_group": "openai",
                "custom_llm_provider": "openai",
                "api_key": "key",
                "api_key_alias": None,
                "team_id": None,
                "team_alias": None,
                "user_id": None,
                "user_email": None,
            }
        ]
    )
    result = _explode_by_sku(frame)
    assert result.height == 1
    assert result["_sku_type"][0] == "prompt_token"


def test_explode_all_zero_tokens_returns_empty_frame():
    # Zero-token requests produce no SKU rows — callers must not emit phantom
    # null-SKU records (ServiceSubcategory=null) into the FOCUS output.
    frame = _frame(_make_row(prompt_tokens=0, completion_tokens=0))
    result = _explode_by_sku(frame)

    assert result.is_empty()
    assert "_sku_type" in result.columns
    assert "_sku_quantity" in result.columns


def test_explode_multiple_input_rows():
    frame = _frame(
        _make_row(prompt_tokens=10, completion_tokens=5),
        _make_row(prompt_tokens=20, completion_tokens=10),
    )
    result = _explode_by_sku(frame)
    assert result.height == 4


# ---------------------------------------------------------------------------
# FocusSkuTransformer tests
# ---------------------------------------------------------------------------


def test_sku_transformer_empty_frame_returns_schema_frame():
    result = FocusSkuTransformer().transform(pl.DataFrame())
    assert result.is_empty()


def test_sku_transformer_all_zero_tokens_returns_empty_schema_frame():
    """End-to-end: a request where all token counts are zero must produce no
    output rows — not a phantom row with ServiceSubcategory=null."""
    frame = _frame(_make_row(prompt_tokens=0, completion_tokens=0,
                             cache_read_input_tokens=0, cache_creation_input_tokens=0))
    result = FocusSkuTransformer().transform(frame)

    assert result.is_empty()
    # Schema must still be correct so downstream serializers don't break
    assert "ServiceSubcategory" in result.columns
    assert "BilledCost" in result.columns


def test_sku_transformer_produces_one_row_per_active_sku():
    frame = _frame(_make_row(prompt_tokens=100, completion_tokens=50))
    result = FocusSkuTransformer().transform(frame)

    assert result.height == 2
    sku_types = set(result["ServiceSubcategory"].to_list())
    assert sku_types == {"prompt_token", "completion_token"}


def test_sku_transformer_service_subcategory_reflects_sku_type():
    frame = _frame(
        _make_row(
            prompt_tokens=10,
            completion_tokens=5,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=2,
        )
    )
    result = FocusSkuTransformer().transform(frame)

    expected = {
        "prompt_token",
        "completion_token",
        "cache_read_token",
        "cache_creation_token",
    }
    assert set(result["ServiceSubcategory"].to_list()) == expected


def test_sku_transformer_consumed_quantity_matches_token_count():
    frame = _frame(_make_row(prompt_tokens=80, completion_tokens=20))
    result = FocusSkuTransformer().transform(frame)

    qty_by_sku = {
        row["ServiceSubcategory"]: row["ConsumedQuantity"]
        for row in result.iter_rows(named=True)
    }
    assert float(qty_by_sku["prompt_token"]) == pytest.approx(80.0)
    assert float(qty_by_sku["completion_token"]) == pytest.approx(20.0)


def test_sku_transformer_consumed_unit_is_token():
    frame = _frame(_make_row(prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "token" for v in result["ConsumedUnit"].to_list())


def test_sku_transformer_pricing_unit_is_token():
    frame = _frame(_make_row(prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "token" for v in result["PricingUnit"].to_list())


def test_sku_transformer_billed_cost_sums_to_total_spend():
    """Sum of BilledCost across all SKU rows must equal the original spend."""
    spend = 0.006
    frame = _frame(_make_row(prompt_tokens=100, completion_tokens=50, spend=spend))
    result = FocusSkuTransformer().transform(frame)

    total = sum(float(v) for v in result["BilledCost"].to_list())
    assert total == pytest.approx(spend, rel=1e-6)


def test_sku_transformer_billed_cost_proportional_to_token_share():
    """Prompt uses 2/3 of tokens → should account for ~2/3 of spend."""
    frame = _frame(_make_row(prompt_tokens=200, completion_tokens=100, spend=0.03))
    result = FocusSkuTransformer().transform(frame)

    costs = {
        row["ServiceSubcategory"]: float(row["BilledCost"])
        for row in result.iter_rows(named=True)
    }
    assert costs["prompt_token"] == pytest.approx(0.02, rel=1e-6)
    assert costs["completion_token"] == pytest.approx(0.01, rel=1e-6)


def test_sku_transformer_resource_id_is_api_key():
    frame = _frame(_make_row(api_key="sk-test-key", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "sk-test-key" for v in result["ResourceId"].to_list())


def test_sku_transformer_resource_name_is_api_key_alias():
    frame = _frame(
        _make_row(api_key_alias="prod-key", prompt_tokens=10, completion_tokens=5)
    )
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "prod-key" for v in result["ResourceName"].to_list())


def test_sku_transformer_resource_type_is_api_key_literal():
    frame = _frame(_make_row(prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "API Key" for v in result["ResourceType"].to_list())


def test_sku_transformer_service_name_is_model_group():
    frame = _frame(_make_row(model_group="anthropic", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "anthropic" for v in result["ServiceName"].to_list())


def test_sku_transformer_charge_description_is_model():
    frame = _frame(_make_row(model="claude-3-5-sonnet", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "claude-3-5-sonnet" for v in result["ChargeDescription"].to_list())


def test_sku_transformer_provider_name_is_custom_llm_provider():
    frame = _frame(_make_row(custom_llm_provider="anthropic", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "anthropic" for v in result["ProviderName"].to_list())


def test_sku_transformer_tags_contain_user_id():
    import json

    frame = _frame(_make_row(user_id="u-123", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)

    for tags_str in result["Tags"].to_list():
        tags = json.loads(tags_str)
        assert tags.get("user_id") == "u-123"


def test_sku_transformer_tags_contain_request_id_when_present():
    """request_id from SpendLogs should appear in Tags when the column exists."""
    import json

    row = {**_make_row(prompt_tokens=10, completion_tokens=5), "request_id": "req-xyz"}
    frame = pl.DataFrame([row])
    result = FocusSkuTransformer().transform(frame)

    for tags_str in result["Tags"].to_list():
        tags = json.loads(tags_str)
        assert tags.get("request_id") == "req-xyz"


def test_default_transformer_tags_omit_request_id_when_absent():
    """DailyUserSpend rows have no request_id — Tags must not contain it."""
    import json

    frame = _frame(_make_row())  # no request_id column
    result = FocusTransformer().transform(frame)

    for tags_str in result["Tags"].to_list():
        tags = json.loads(tags_str)
        assert "request_id" not in tags


def test_sku_transformer_tags_contain_team_id():
    import json

    frame = _frame(_make_row(team_id="team-abc", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)

    for tags_str in result["Tags"].to_list():
        tags = json.loads(tags_str)
        assert tags.get("team_id") == "team-abc"


def test_sku_transformer_charge_period_start_from_date():
    frame = _frame(_make_row(date="2024-06-01", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v.startswith("2024-06-01") for v in result["ChargePeriodStart"].to_list())


def test_sku_transformer_charge_period_end_is_next_day():
    frame = _frame(_make_row(date="2024-06-01", prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v.startswith("2024-06-02") for v in result["ChargePeriodEnd"].to_list())


def test_sku_transformer_billing_currency_is_usd():
    frame = _frame(_make_row(prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "USD" for v in result["BillingCurrency"].to_list())


def test_sku_transformer_service_category_is_ai_ml():
    frame = _frame(_make_row(prompt_tokens=10, completion_tokens=5))
    result = FocusSkuTransformer().transform(frame)
    assert all(v == "AI and Machine Learning" for v in result["ServiceCategory"].to_list())


def test_sku_transformer_multiple_input_rows_all_expanded():
    frame = _frame(
        _make_row(prompt_tokens=10, completion_tokens=5),
        _make_row(prompt_tokens=20, completion_tokens=10),
    )
    result = FocusSkuTransformer().transform(frame)
    assert result.height == 4


def test_sku_transformer_spend_zero_produces_zero_cost():
    frame = _frame(_make_row(prompt_tokens=100, completion_tokens=50, spend=0.0))
    result = FocusSkuTransformer().transform(frame)
    assert all(float(v) == 0.0 for v in result["BilledCost"].to_list())


def test_sku_transformer_four_sku_costs_sum_to_total():
    """With all four token types active, sum of costs still equals total spend."""
    frame = _frame(
        _make_row(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
            spend=0.1,
        )
    )
    result = FocusSkuTransformer().transform(frame)

    total = sum(float(v) for v in result["BilledCost"].to_list())
    assert total == pytest.approx(0.1, rel=1e-6)


def test_sku_transformer_cache_only_tokens():
    """Edge case: only cache tokens, no prompt/completion."""
    frame = _frame(
        _make_row(
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_input_tokens=40,
            cache_creation_input_tokens=60,
            spend=0.005,
        )
    )
    result = FocusSkuTransformer().transform(frame)

    sku_types = set(result["ServiceSubcategory"].to_list())
    assert sku_types == {"cache_read_token", "cache_creation_token"}
    assert result.height == 2
    total = sum(float(v) for v in result["BilledCost"].to_list())
    assert total == pytest.approx(0.005, rel=1e-6)
