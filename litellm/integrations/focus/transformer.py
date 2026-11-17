"""Focus export data transformer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import timedelta
from typing import Final

import polars as pl

from .schema import FOCUS_NORMALIZED_SCHEMA

_TAG_KEYS: Final = (
    "team_id",
    "team_alias",
    "organization_id",
    "organization_alias",
    "user_id",
    "user_email",
    "api_key_alias",
    "model",
    "model_group",
    "custom_llm_provider",
)


def _build_tags_expr(available_keys: list[str]) -> pl.Expr:
    """Build a Polars expression that produces a JSON Tags string per row.

    Uses ``pl.struct`` + ``map_elements`` to avoid materialising the entire
    DataFrame to a list of Python dicts.  The JSON serialisation callback
    still runs in Python (GIL-bound), but struct-packing and loop dispatch
    are handled by Polars' Rust engine.
    """

    def _struct_to_json(row: Mapping[str, object]) -> str:
        tags: Final = {k: str(v) for k, v in row.items() if v is not None}
        return json.dumps(tags) if tags else "{}"

    return pl.struct(available_keys).map_elements(_struct_to_json, return_dtype=pl.String).alias("Tags")


def _requester_details(row: Mapping[str, object]) -> str | None:
    principal: Final = (
        (
            {
                "key": "Principal",
                "value": {
                    "Type": "User",
                    **{name: row["user_email"] for name in ("Name", "Email") if row["user_email"] is not None},
                },
            },
        )
        if row["user_id"] is not None
        else ()
    )
    credential: Final = (
        (
            {
                "key": "Credential",
                "value": {
                    "Type": "API Key",
                    **({"Name": row["api_key_alias"]} if row["api_key_alias"] is not None else {}),
                },
            },
        )
        if row["api_key"] is not None
        else ()
    )
    entries: Final = (*principal, *credential)
    return json.dumps(entries) if entries else None


def _sku_price_details(row: Mapping[str, object]) -> str | None:
    if row["custom_llm_provider"] is None or row["model"] is None:
        return None
    return json.dumps({"ModelId": row["model"]})


class FocusTransformer:
    """Transforms LiteLLM DB rows into Focus-compatible schema."""

    schema: Final = FOCUS_NORMALIZED_SCHEMA

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a normalized frame expected by downstream serializers."""
        if frame.is_empty():
            return pl.DataFrame(schema=self.schema)

        available_keys: Final = [k for k in _TAG_KEYS if k in frame.columns]
        tagged: Final = frame.with_columns(
            _build_tags_expr(available_keys) if available_keys else pl.lit("{}").alias("Tags"),
            *(
                pl.lit(None, dtype=pl.String).alias(name)
                for name in ("user_id", "user_email")
                if name not in frame.columns
            ),
        )
        dated: Final = tagged.with_columns(
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime(time_unit="us"), format="%Y-%m-%d", strict=False)
            .alias("usage_date"),
        )
        periods: Final = dated.with_columns(
            pl.col("usage_date").alias("ChargePeriodStart"),
            (pl.col("usage_date") + timedelta(days=1)).alias("ChargePeriodEnd"),
        )

        def fmt(col: pl.Expr) -> pl.Expr:
            return col.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        DEC: Final = pl.Decimal(18, 6)

        def dec(col: pl.Expr) -> pl.Expr:
            return col.cast(DEC)

        def string_col(name: str) -> pl.Expr:
            return pl.col(name).cast(pl.String)

        none_str: Final = pl.lit(None, dtype=pl.Utf8)
        none_dec: Final = pl.lit(None, dtype=pl.Decimal(18, 6))
        spend: Final = dec(pl.col("spend").fill_null(0.0))
        requests: Final = dec(pl.col("api_requests").cast(pl.Int64).cast(pl.Float64).fill_null(0.0))
        sku_id: Final = pl.concat_str(string_col("custom_llm_provider"), string_col("model"), separator="/")

        return periods.select(
            spend.alias("BilledCost"),
            string_col("api_key").alias("BillingAccountId"),
            string_col("api_key_alias").alias("BillingAccountName"),
            pl.when(pl.col("api_key").is_not_null())
            .then(pl.lit("API Key"))
            .otherwise(none_str)
            .alias("BillingAccountType"),
            pl.lit("USD").alias("BillingCurrency"),
            fmt(pl.col("ChargePeriodEnd")).alias("BillingPeriodEnd"),
            fmt(pl.col("ChargePeriodStart")).alias("BillingPeriodStart"),
            pl.lit("Usage").alias("ChargeCategory"),
            none_str.alias("ChargeClass"),
            string_col("model").alias("ChargeDescription"),
            pl.lit("Usage-Based").alias("ChargeFrequency"),
            fmt(pl.col("ChargePeriodEnd")).alias("ChargePeriodEnd"),
            fmt(pl.col("ChargePeriodStart")).alias("ChargePeriodStart"),
            requests.alias("ConsumedQuantity"),
            pl.lit("Requests").alias("ConsumedUnit"),
            spend.alias("ContractedCost"),
            none_dec.alias("ContractedUnitPrice"),
            string_col("api_key").alias("CredentialId"),
            spend.alias("EffectiveCost"),
            string_col("custom_llm_provider").alias("InvoiceIssuerName"),
            none_str.alias("InvoiceId"),
            spend.alias("ListCost"),
            none_dec.alias("ListUnitPrice"),
            none_str.alias("AvailabilityZone"),
            pl.lit("USD").alias("PricingCurrency"),
            none_str.alias("PricingCategory"),
            requests.alias("PricingQuantity"),
            none_dec.alias("PricingCurrencyContractedUnitPrice"),
            spend.alias("PricingCurrencyEffectiveCost"),
            none_dec.alias("PricingCurrencyListUnitPrice"),
            pl.lit("Requests").alias("PricingUnit"),
            string_col("user_id").alias("PrincipalId"),
            string_col("custom_llm_provider").alias("ServiceProviderName"),
            string_col("custom_llm_provider").alias("HostProviderName"),
            pl.struct("user_id", "user_email", "api_key", "api_key_alias")
            .map_elements(_requester_details, return_dtype=pl.String)
            .alias("RequesterDetails"),
            none_str.alias("RegionId"),
            none_str.alias("RegionName"),
            string_col("model").alias("ResourceId"),
            string_col("model").alias("ResourceName"),
            string_col("model").alias("ResourceType"),
            pl.lit("AI and Machine Learning").alias("ServiceCategory"),
            pl.lit("Generative AI").alias("ServiceSubcategory"),
            string_col("model_group").alias("ServiceName"),
            sku_id.alias("SkuId"),
            pl.concat_str(sku_id, pl.lit("requests"), separator="/").alias("SkuPriceId"),
            pl.struct("custom_llm_provider", "model")
            .map_elements(_sku_price_details, return_dtype=pl.String)
            .alias("SkuPriceDetails"),
            string_col("team_id").alias("SubAccountId"),
            string_col("team_alias").alias("SubAccountName"),
            pl.when(pl.col("team_id").is_not_null()).then(pl.lit("Team")).otherwise(none_str).alias("SubAccountType"),
            string_col("Tags").alias("Tags"),
        ).select(self.schema.names())
