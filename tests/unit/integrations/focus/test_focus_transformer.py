"""Tests for FocusTransformer — ConsumedQuantity / PricingQuantity correctness."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from litellm.integrations.focus.transformer import FocusTransformer


def _base_row(**overrides) -> dict:
    row = {
        "date": "2026-05-25",
        "user_id": "u1",
        "api_key": "sk-test",
        "api_key_alias": "my-key",
        "model": "gpt-4o",
        "model_group": "openai",
        "custom_llm_provider": "openai",
        "spend": 0.05,
        "api_requests": 3,
        "team_id": "team1",
        "team_alias": "Engineering",
        "user_email": "user@example.com",
    }
    row.update(overrides)
    return row


def _transform(rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return FocusTransformer().transform(frame)


def test_consumed_quantity_reflects_api_requests():
    result = _transform([_base_row(api_requests=7)])
    assert result["ConsumedQuantity"][0] == Decimal("7.000000")


def test_pricing_quantity_reflects_api_requests():
    result = _transform([_base_row(api_requests=7)])
    assert result["PricingQuantity"][0] == Decimal("7.000000")


def test_null_api_requests_falls_back_to_zero_not_one():
    """Rows with NULL api_requests (old schema rows) must produce 0, not 1."""
    result = _transform([_base_row(api_requests=None)])
    assert result["ConsumedQuantity"][0] == Decimal("0.000000")
    assert result["PricingQuantity"][0] == Decimal("0.000000")


def test_zero_api_requests_stays_zero():
    result = _transform([_base_row(api_requests=0)])
    assert result["ConsumedQuantity"][0] == Decimal("0.000000")
    assert result["PricingQuantity"][0] == Decimal("0.000000")


def test_bigint_api_requests_cast_correctly():
    """api_requests comes from Postgres as BigInt — large values must not overflow."""
    result = _transform([_base_row(api_requests=1_000_000)])
    assert result["ConsumedQuantity"][0] == Decimal("1000000.000000")
    assert result["PricingQuantity"][0] == Decimal("1000000.000000")


def test_consumed_and_pricing_quantity_match():
    """ConsumedQuantity and PricingQuantity must always be equal."""
    result = _transform([_base_row(api_requests=42)])
    assert result["ConsumedQuantity"][0] == result["PricingQuantity"][0]
