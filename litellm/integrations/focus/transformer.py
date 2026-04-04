"""Focus export data transformer."""

from __future__ import annotations

import json
from datetime import timedelta

import polars as pl

from litellm._logging import verbose_logger

from .schema import FOCUS_NORMALIZED_SCHEMA


_TAG_KEYS = (
    "request_id",
    "team_id",
    "team_alias",
    "user_id",
    "user_email",
    "api_key_alias",
    "model",
    "model_group",
    "custom_llm_provider",
)

# Token columns present in LiteLLM_DailyUserSpend and their FOCUS SKU names.
# Used by FocusSkuTransformer to explode each DB row into one row per active
# token type (prompt_token, completion_token, cache_read_token,
# cache_creation_token).
_SKU_COLUMNS: list[tuple[str, str]] = [
    ("prompt_tokens", "prompt_token"),
    ("completion_tokens", "completion_token"),
    ("cache_read_input_tokens", "cache_read_token"),
    ("cache_creation_input_tokens", "cache_creation_token"),
]


def _build_tags_expr(available_keys: list[str]) -> pl.Expr:
    """Build a Polars expression that produces a JSON Tags string per row.

    Uses ``pl.struct`` + ``map_elements`` to avoid materialising the entire
    DataFrame to a list of Python dicts.  The JSON serialisation callback
    still runs in Python (GIL-bound), but struct-packing and loop dispatch
    are handled by Polars' Rust engine.
    """

    def _struct_to_json(row: dict) -> str:
        tags = {k: str(v) for k, v in row.items() if v is not None}
        return json.dumps(tags) if tags else "{}"

    return (
        pl.struct(available_keys)
        .map_elements(_struct_to_json, return_dtype=pl.String)
        .alias("Tags")
    )


def _explode_by_sku(frame: pl.DataFrame) -> pl.DataFrame:
    """Explode each row into one row per active SKU type (token category).

    Adds two new columns to the returned frame:

    - ``_sku_type``: token category label (e.g. ``"prompt_token"``)
    - ``_sku_quantity``: token count for that SKU type (Float64)

    Rows where a given token column is zero or null are excluded from that
    SKU's sub-frame.  If the incoming frame is empty, or all token counts are
    zero, an empty frame (0 rows) is returned so callers don't emit phantom
    zero-cost records with a null SKU type into the FOCUS output.
    """
    if frame.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("_sku_type"),
            pl.lit(None, dtype=pl.Float64).alias("_sku_quantity"),
        )

    sku_frames: list[pl.DataFrame] = []
    for col_name, sku_name in _SKU_COLUMNS:
        if col_name not in frame.columns:
            continue
        filtered = frame.filter(pl.col(col_name).fill_null(0) > 0).with_columns(
            pl.col(col_name).cast(pl.Float64).alias("_sku_quantity"),
            pl.lit(sku_name).alias("_sku_type"),
        )
        if not filtered.is_empty():
            sku_frames.append(filtered)

    if not sku_frames:
        # All token counts are zero — return an empty frame so no phantom
        # null-SKU rows reach the FOCUS output (ServiceSubcategory would be null,
        # which downstream billing consumers may reject or mishandle).
        return frame.head(0).with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("_sku_type"),
            pl.lit(None, dtype=pl.Float64).alias("_sku_quantity"),
        )

    return pl.concat(sku_frames)


class FocusTransformer:
    """Transforms LiteLLM DB rows into Focus-compatible schema."""

    schema = FOCUS_NORMALIZED_SCHEMA

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a normalized frame expected by downstream serializers."""
        if frame.is_empty():
            return pl.DataFrame(schema=self.schema)

        # Build Tags JSON from metadata columns using vectorized Polars expression
        available_keys = [k for k in _TAG_KEYS if k in frame.columns]
        if available_keys:
            frame = frame.with_columns(_build_tags_expr(available_keys))
        else:
            frame = frame.with_columns(pl.lit("{}").alias("Tags"))

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
            pl.col("Tags").cast(pl.String).alias("Tags"),
        )


class FocusSkuTransformer:
    """Transforms LiteLLM DB rows into Focus-compatible schema, one row per SKU.

    Each input row is exploded into one output row per active token type
    (prompt_token, completion_token, cache_read_token, cache_creation_token).
    Cost is allocated proportionally across SKUs by token count so that the
    sum of BilledCost across all SKU rows for a request equals the original
    total spend.

    This produces billing line-items that map directly to FOCUS / vendor SKU
    conventions, where each row represents a single charge type:

        ProviderName | ServiceName | ServiceSubcategory | ResourceId | BilledCost | ConsumedQuantity | ConsumedUnit
        openai       | openai      | prompt_token       | sk-...     | 0.002      | 100              | token
        openai       | openai      | completion_token   | sk-...     | 0.001      | 50               | token
    """

    schema = FOCUS_NORMALIZED_SCHEMA

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a normalized, SKU-exploded frame for downstream serializers."""
        if frame.is_empty():
            return pl.DataFrame(schema=self.schema)

        # Explode each DB row into one row per active SKU type.
        # Returns an empty frame when all token counts are zero.
        frame = _explode_by_sku(frame)
        if frame.is_empty():
            return pl.DataFrame(schema=self.schema)

        # Proportional cost per SKU: spend * (sku_tokens / total_tokens).
        # This ensures cost rows sum back to the original request spend.
        present_token_cols = [c for c, _ in _SKU_COLUMNS if c in frame.columns]
        if present_token_cols:
            total_tokens_expr = pl.sum_horizontal(
                [pl.col(c).fill_null(0).cast(pl.Float64) for c in present_token_cols]
            )
            frame = frame.with_columns(
                pl.when(total_tokens_expr > 0)
                .then(
                    pl.col("spend").fill_null(0.0)
                    * pl.col("_sku_quantity")
                    / total_tokens_expr
                )
                .otherwise(0.0)
                .alias("_sku_cost"),
            )
        else:
            non_zero_spend = frame.filter(pl.col("spend").fill_null(0.0) > 0)
            if not non_zero_spend.is_empty():
                verbose_logger.warning(
                    "FocusSkuTransformer: frame has non-zero spend but no recognised "
                    "token columns (%s). Cost will be zeroed in the FOCUS output. "
                    "Ensure the data source includes at least one of: %s.",
                    list(frame.columns),
                    [c for c, _ in _SKU_COLUMNS],
                )
            frame = frame.with_columns(pl.lit(0.0).alias("_sku_cost"))

        # Build Tags JSON from metadata columns
        available_keys = [k for k in _TAG_KEYS if k in frame.columns]
        if available_keys:
            frame = frame.with_columns(_build_tags_expr(available_keys))
        else:
            frame = frame.with_columns(pl.lit("{}").alias("Tags"))

        # Derive period start/end from usage date
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

        def fmt(col: pl.Expr) -> pl.Expr:
            return col.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        DEC = pl.Decimal(18, 6)

        def dec(col: pl.Expr) -> pl.Expr:
            return col.cast(DEC)

        none_str = pl.lit(None, dtype=pl.Utf8)
        none_dec = pl.lit(None, dtype=pl.Decimal(18, 6))

        return frame.select(
            dec(pl.col("_sku_cost").fill_null(0.0)).alias("BilledCost"),
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
            pl.col("_sku_quantity").cast(DEC).alias("ConsumedQuantity"),
            pl.lit("token").alias("ConsumedUnit"),
            dec(pl.col("_sku_cost").fill_null(0.0)).alias("ContractedCost"),
            none_str.alias("ContractedUnitPrice"),
            dec(pl.col("_sku_cost").fill_null(0.0)).alias("EffectiveCost"),
            pl.col("custom_llm_provider").cast(pl.String).alias("InvoiceIssuerName"),
            none_str.alias("InvoiceId"),
            dec(pl.col("_sku_cost").fill_null(0.0)).alias("ListCost"),
            none_dec.alias("ListUnitPrice"),
            none_str.alias("AvailabilityZone"),
            pl.lit("USD").alias("PricingCurrency"),
            none_str.alias("PricingCategory"),
            pl.col("_sku_quantity").cast(DEC).alias("PricingQuantity"),
            none_dec.alias("PricingCurrencyContractedUnitPrice"),
            dec(pl.col("_sku_cost").fill_null(0.0)).alias("PricingCurrencyEffectiveCost"),
            none_dec.alias("PricingCurrencyListUnitPrice"),
            pl.lit("token").alias("PricingUnit"),
            pl.col("custom_llm_provider").cast(pl.String).alias("ProviderName"),
            pl.col("custom_llm_provider").cast(pl.String).alias("PublisherName"),
            none_str.alias("RegionId"),
            none_str.alias("RegionName"),
            pl.col("api_key").cast(pl.String).alias("ResourceId"),
            pl.col("api_key_alias").cast(pl.String).alias("ResourceName"),
            pl.lit("API Key").alias("ResourceType"),
            pl.lit("AI and Machine Learning").alias("ServiceCategory"),
            pl.col("_sku_type").cast(pl.String).alias("ServiceSubcategory"),
            pl.col("model_group").cast(pl.String).alias("ServiceName"),
            pl.col("team_id").cast(pl.String).alias("SubAccountId"),
            pl.col("team_alias").cast(pl.String).alias("SubAccountName"),
            none_str.alias("SubAccountType"),
            pl.col("Tags").cast(pl.String).alias("Tags"),
        )
