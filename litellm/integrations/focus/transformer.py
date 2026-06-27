"""Focus export data transformer."""

from __future__ import annotations

import json
from datetime import timedelta

import polars as pl

from .schema import FOCUS_NORMALIZED_SCHEMA

_TAG_KEYS = (
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

_REQUEST_TAGS_KEY = "request_tags"
_REQUEST_TAGS_TRUNCATED_KEY = "request_tags_truncated"

# Request tags are caller-controlled and unbounded. Cap how many (and how long)
# land in a single row's Tags blob so it cannot exceed a destination's per-row
# size limit (Vantage drops any row over 2 MB). The cap is on characters; even
# the worst case (64 tags x 128 multibyte chars, double-JSON-encoded then CSV-
# escaped) is on the order of ~115 KB, well under 2 MB.
_MAX_REQUEST_TAGS = 64
_MAX_TAG_LENGTH = 128


def _build_tags_expr(tag_columns: list[str]) -> pl.Expr:
    """Build a Polars expression that produces a JSON Tags string per row.

    Uses ``pl.struct`` + ``map_elements`` to avoid materialising the entire
    DataFrame to a list of Python dicts.  The JSON serialisation callback
    still runs in Python (GIL-bound), but struct-packing and loop dispatch
    are handled by Polars' Rust engine.

    Request-level tags arrive as a list column and are encoded as a JSON
    array string so the Tags map stays a flat string-to-string object. The
    list is capped to a bounded count/length so a flood of caller-supplied
    tags cannot push the row past a destination's per-row size limit; when
    that happens a ``request_tags_truncated`` marker carries the true count.
    """

    def _struct_to_json(row: dict[str, object]) -> str:
        metadata = {
            k: str(v)
            for k, v in row.items()
            if k != _REQUEST_TAGS_KEY and v is not None
        }
        raw_tags = row.get(_REQUEST_TAGS_KEY)
        if not raw_tags or not isinstance(raw_tags, (list, tuple)):
            return json.dumps(metadata) if metadata else "{}"
        total = len(raw_tags)
        capped: list[str] = [
            str(t)[:_MAX_TAG_LENGTH] for t in raw_tags[:_MAX_REQUEST_TAGS]
        ]
        tags = {
            **metadata,
            _REQUEST_TAGS_KEY: json.dumps(capped),
            **(
                {_REQUEST_TAGS_TRUNCATED_KEY: str(total)}
                if total > _MAX_REQUEST_TAGS
                else {}
            ),
        }
        return json.dumps(tags)

    return (
        pl.struct(tag_columns)
        .map_elements(_struct_to_json, return_dtype=pl.String)
        .alias("Tags")
    )


class FocusTransformer:
    """Transforms LiteLLM DB rows into Focus-compatible schema."""

    schema = FOCUS_NORMALIZED_SCHEMA

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a normalized frame expected by downstream serializers."""
        if frame.is_empty():
            return pl.DataFrame(schema=self.schema)

        # Build Tags JSON from metadata columns using vectorized Polars expression
        tag_columns = [k for k in (*_TAG_KEYS, _REQUEST_TAGS_KEY) if k in frame.columns]
        if tag_columns:
            frame = frame.with_columns(_build_tags_expr(tag_columns))
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
            dec(
                pl.col("api_requests").cast(pl.Int64).cast(pl.Float64).fill_null(0.0)
            ).alias("ConsumedQuantity"),
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
            dec(
                pl.col("api_requests").cast(pl.Int64).cast(pl.Float64).fill_null(0.0)
            ).alias("PricingQuantity"),
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
