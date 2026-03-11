"""Focus export data transformer."""

from __future__ import annotations

import json
from datetime import timedelta

import polars as pl

from .schema import FOCUS_NORMALIZED_SCHEMA


def _build_tags_json(row: dict) -> str:
    """Build a JSON string of metadata tags from a DB row.

    Vantage uses this for Token Allocation — enriching billing data with
    team, user, and API key metadata.
    """
    tags: dict[str, str] = {}
    for key in (
        "team_id",
        "team_alias",
        "user_id",
        "user_email",
        "api_key_alias",
        "model",
        "model_group",
        "custom_llm_provider",
    ):
        val = row.get(key)
        if val is not None:
            tags[key] = str(val)
    return json.dumps(tags) if tags else "{}"


class FocusTransformer:
    """Transforms LiteLLM DB rows into Focus-compatible schema."""

    schema = FOCUS_NORMALIZED_SCHEMA

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a normalized frame expected by downstream serializers."""
        if frame.is_empty():
            return pl.DataFrame(schema=self.schema)

        # Build Tags JSON from metadata columns
        tag_col = "Tags"
        tag_keys = [
            "team_id", "team_alias", "user_id", "user_email",
            "api_key_alias", "model", "model_group", "custom_llm_provider",
        ]
        available_keys = [k for k in tag_keys if k in frame.columns]
        if available_keys:
            tags_series = frame.select(available_keys).to_dicts()
            tags_json = [_build_tags_json(row) for row in tags_series]
            frame = frame.with_columns(pl.Series(tag_col, tags_json))
        else:
            frame = frame.with_columns(pl.lit("{}").alias(tag_col))

        # derive period start/end from usage date
        frame = frame.with_columns(
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime(time_unit="us"), format="%Y-%m-%d", strict=False)
            .alias("usage_date"),
        )
        frame = frame.with_columns(
            pl.col("usage_date").alias("ChargePeriodStart"),
            (pl.col("usage_date") + timedelta(days=1)).alias("ChargePeriodEnd"),
        )

        def fmt(col):
            return col.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        DEC = pl.Decimal(18, 6)

        def dec(col):
            return col.cast(DEC)

        none_str = pl.lit(None, dtype=pl.Utf8)
        none_dec = pl.lit(None, dtype=pl.Decimal(18, 6))

        return frame.select(
            dec(pl.col("spend").fill_null(0.0)).alias("BilledCost"),
            pl.col("api_key").cast(pl.String).alias("BillingAccountId"),
            pl.col("api_key_alias").cast(pl.String).alias("BillingAccountName"),
            pl.lit("API Key").alias("BillingAccountType"),
            pl.lit("USD").alias("BillingCurrency"),
            fmt(pl.col("ChargePeriodEnd")).alias("BillingPeriodEnd"),
            fmt(pl.col("ChargePeriodStart")).alias("BillingPeriodStart"),
            pl.lit("Usage").alias("ChargeCategory"),
            none_str.alias("ChargeClass"),
            pl.col("model").cast(pl.String).alias("ChargeDescription"),
            pl.lit("Usage-Based").alias("ChargeFrequency"),
            fmt(pl.col("ChargePeriodEnd")).alias("ChargePeriodEnd"),
            fmt(pl.col("ChargePeriodStart")).alias("ChargePeriodStart"),
            dec(pl.lit(1.0)).alias("ConsumedQuantity"),
            pl.lit("Requests").alias("ConsumedUnit"),
            dec(pl.col("spend").fill_null(0.0)).alias("ContractedCost"),
            none_str.alias("ContractedUnitPrice"),
            dec(pl.col("spend").fill_null(0.0)).alias("EffectiveCost"),
            pl.col("custom_llm_provider").cast(pl.String).alias("InvoiceIssuerName"),
            none_str.alias("InvoiceId"),
            dec(pl.col("spend").fill_null(0.0)).alias("ListCost"),
            none_dec.alias("ListUnitPrice"),
            none_str.alias("AvailabilityZone"),
            pl.lit("USD").alias("PricingCurrency"),
            none_str.alias("PricingCategory"),
            dec(pl.lit(1.0)).alias("PricingQuantity"),
            none_dec.alias("PricingCurrencyContractedUnitPrice"),
            dec(pl.col("spend").fill_null(0.0)).alias("PricingCurrencyEffectiveCost"),
            none_dec.alias("PricingCurrencyListUnitPrice"),
            pl.lit("Requests").alias("PricingUnit"),
            pl.col("custom_llm_provider").cast(pl.String).alias("ProviderName"),
            pl.col("custom_llm_provider").cast(pl.String).alias("PublisherName"),
            none_str.alias("RegionId"),
            none_str.alias("RegionName"),
            pl.col("model").cast(pl.String).alias("ResourceId"),
            pl.col("model").cast(pl.String).alias("ResourceName"),
            pl.col("model").cast(pl.String).alias("ResourceType"),
            pl.lit("AI and Machine Learning").alias("ServiceCategory"),
            pl.lit("Generative AI").alias("ServiceSubcategory"),
            pl.col("model_group").cast(pl.String).alias("ServiceName"),
            pl.col("team_id").cast(pl.String).alias("SubAccountId"),
            pl.col("team_alias").cast(pl.String).alias("SubAccountName"),
            none_str.alias("SubAccountType"),
            pl.col(tag_col).cast(pl.String).alias("Tags"),
        )
