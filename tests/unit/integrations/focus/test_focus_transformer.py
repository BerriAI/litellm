"""Tests for FocusTransformer — ConsumedQuantity / PricingQuantity correctness."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Final

import jsonschema
import polars as pl
import pytest

from litellm.integrations.focus.destinations.vantage_destination import _strip_unsupported_columns
from litellm.integrations.focus.schema import FOCUS_NORMALIZED_SCHEMA
from litellm.integrations.focus.serializers.csv import FocusCsvSerializer
from litellm.integrations.focus.transformer import FocusTransformer


def _base_row(**overrides: object) -> dict[str, object]:
    return {
        "date": "2026-05-25",
        "user_id": "u1",
        "api_key": "a" * 64,
        "api_key_alias": "my-key",
        "model": "gpt-4o",
        "model_group": "openai",
        "custom_llm_provider": "openai",
        "spend": 0.05,
        "api_requests": 3,
        "team_id": "team1",
        "team_alias": "Engineering",
        "user_email": "user@example.com",
        **overrides,
    }


def _transform(rows: list[dict[str, object]]) -> pl.DataFrame:
    frame: Final = pl.DataFrame(rows, infer_schema_length=None)
    return FocusTransformer().transform(frame)


def test_consumed_quantity_reflects_api_requests():
    result = _transform([_base_row(api_requests=7)])
    assert result["ConsumedQuantity"][0] == Decimal("7.000000")


@pytest.mark.parametrize("requests", [7, 0, None])
def test_aggregate_request_meter_preserves_request_quantities(requests: int | None) -> None:
    result: Final = _transform([_base_row(api_requests=requests)])
    assert result.select("PricingQuantity", "PricingUnit").row(0) == (Decimal(requests or 0), "Requests")
    assert result.select("PricingQuantity", "PricingUnit").row(0) == result.select(
        "ConsumedQuantity", "ConsumedUnit"
    ).row(0)


@pytest.mark.parametrize("provider", ["openai", "anthropic", None])
@pytest.mark.parametrize("model", ["model-a", 'provider/model-"b"', None])
def test_sku_identity_and_details_require_provider_and_model(provider: str | None, model: str | None) -> None:
    result: Final = _transform([_base_row(custom_llm_provider=provider, model=model)])
    expected_sku: Final = f"{provider}/{model}" if provider is not None and model is not None else None
    assert result.select("SkuId", "SkuPriceId").row(0) == (
        expected_sku,
        f"{expected_sku}/requests" if expected_sku is not None else None,
    )
    details: Final = result["SkuPriceDetails"][0]
    if expected_sku is None:
        assert details is None
        return
    assert json.loads(details) == {"ModelId": model}


def test_sku_identity_is_independent_of_account_date_and_price() -> None:
    result: Final = _transform(
        [
            _base_row(),
            _base_row(api_key="b" * 64, spend=0.1, date="2026-05-26"),
        ]
    )
    assert result.select("SkuId", "SkuPriceId", "SkuPriceDetails").row(0) == result.select(
        "SkuId", "SkuPriceId", "SkuPriceDetails"
    ).row(1)
    assert result["SkuPriceId"][0] is not None


def test_null_api_requests_falls_back_to_zero_not_one():
    """Rows with NULL api_requests (old schema rows) must produce 0, not 1."""
    result = _transform([_base_row(api_requests=None)])
    assert result["ConsumedQuantity"][0] == Decimal("0.000000")


def test_zero_api_requests_stays_zero():
    result = _transform([_base_row(api_requests=0)])
    assert result["ConsumedQuantity"][0] == Decimal("0.000000")


def test_bigint_api_requests_cast_correctly():
    """api_requests comes from Postgres as BigInt — large values must not overflow."""
    result = _transform([_base_row(api_requests=1_000_000)])
    assert result["ConsumedQuantity"][0] == Decimal("1000000.000000")


def test_empty_and_populated_exports_have_the_same_schema():
    empty: Final = FocusTransformer().transform(pl.DataFrame())
    result: Final = _transform([_base_row()])
    assert empty.schema == result.schema == FOCUS_NORMALIZED_SCHEMA
    assert result.columns == FOCUS_NORMALIZED_SCHEMA.names()
    assert "ProviderName" not in result.columns
    assert "PublisherName" not in result.columns


@pytest.mark.parametrize("user_id", ["u1", None])
@pytest.mark.parametrize("api_key", ["a" * 64, None])
@pytest.mark.parametrize("has_names", [True, False])
def test_requester_entries_only_describe_present_identifiers(
    user_id: str | None, api_key: str | None, has_names: bool
) -> None:
    email: Final = 'person+"qa"@example.com' if has_names else None
    alias: Final = 'production "key" \\ café' if has_names else None
    result: Final = _transform([_base_row(user_id=user_id, api_key=api_key, user_email=email, api_key_alias=alias)])
    assert result.select("PrincipalId", "CredentialId", "BillingAccountType").row(0) == (
        user_id,
        api_key,
        "API Key" if api_key is not None else None,
    )
    expected_principal: Final = (
        [{"key": "Principal", "value": {"Type": "User", **({"Name": email, "Email": email} if has_names else {})}}]
        if user_id is not None
        else []
    )
    expected_credential: Final = (
        [{"key": "Credential", "value": {"Type": "API Key", **({"Name": alias} if has_names else {})}}]
        if api_key is not None
        else []
    )
    raw: Final = result["RequesterDetails"][0]
    if user_id is None and api_key is None:
        assert raw is None
        return
    details: Final = json.loads(raw)
    assert details == expected_principal + expected_credential
    schema: Final = json.loads(Path(__file__).with_name("requesterdetailsobjectschema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(details)


def test_provider_names_and_daily_cost_survive_csv_and_vantage_filtering() -> None:
    source: Final = _base_row(custom_llm_provider="anthropic", api_requests=3)
    result: Final = _transform([source])
    assert result.select("ServiceProviderName", "HostProviderName", "InvoiceIssuerName").row(0) == (
        "anthropic",
        "anthropic",
        "anthropic",
    )
    assert result.select(
        "ChargeCategory", "ChargeFrequency", "ConsumedUnit", "ServiceCategory", "ServiceSubcategory"
    ).row(0) == ("Usage", "Usage-Based", "Requests", "AI and Machine Learning", "Generative AI")
    assert result.height == 1
    assert result["BilledCost"][0] == Decimal("0.050000")
    assert result.select("ResourceId", "ResourceName", "ChargeDescription").row(0) == (source["model"],) * 3
    csv_bytes: Final = FocusCsvSerializer().serialize(result)
    roundtrip: Final = pl.read_csv(csv_bytes)
    assert json.loads(roundtrip["RequesterDetails"][0]) == json.loads(result["RequesterDetails"][0])
    vantage: Final = pl.read_csv(_strip_unsupported_columns(csv_bytes))
    assert vantage["BilledCost"].to_list() == roundtrip["BilledCost"].to_list()
    assert "ServiceProviderName" not in vantage.columns
    assert "HostProviderName" not in vantage.columns
    assert "RequesterDetails" not in vantage.columns
    assert "SkuId" not in vantage.columns
    assert "SkuPriceId" not in vantage.columns
    assert "SkuPriceDetails" not in vantage.columns


def test_missing_optional_user_columns_leave_only_credential_details() -> None:
    source: Final = pl.DataFrame([_base_row()]).drop("user_id", "user_email")
    result: Final = FocusTransformer().transform(source)
    assert result["PrincipalId"][0] is None
    assert json.loads(result["RequesterDetails"][0]) == [
        {"key": "Credential", "value": {"Type": "API Key", "Name": "my-key"}}
    ]


@pytest.mark.parametrize("team_id", ["team1", None])
def test_subaccount_type_matches_team_presence(team_id: str | None) -> None:
    result: Final = _transform([_base_row(team_id=team_id)])
    assert result["SubAccountType"][0] == ("Team" if team_id is not None else None)
