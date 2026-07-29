"""Schema definitions for Focus export data."""

from __future__ import annotations

import polars as pl

# see: https://focus.finops.org/focus-specification/v1-2/
FOCUS_NORMALIZED_SCHEMA = pl.Schema(
    [
        ("BilledCost", pl.Decimal(18, 6)),
        ("BillingAccountId", pl.String),
        ("BillingAccountName", pl.String),
        ("BillingCurrency", pl.String),
        ("BillingPeriodStart", pl.Datetime(time_unit="us")),
        ("BillingPeriodEnd", pl.Datetime(time_unit="us")),
        ("ChargeCategory", pl.String),
        ("ChargeClass", pl.String),
        ("ChargeDescription", pl.String),
        ("ChargeFrequency", pl.String),
        ("ChargePeriodStart", pl.Datetime(time_unit="us")),
        ("ChargePeriodEnd", pl.Datetime(time_unit="us")),
        ("ConsumedQuantity", pl.Decimal(18, 6)),
        ("ConsumedUnit", pl.String),
        ("ContractedCost", pl.Decimal(18, 6)),
        ("ContractedUnitPrice", pl.Decimal(18, 6)),
        ("EffectiveCost", pl.Decimal(18, 6)),
        ("InvoiceIssuerName", pl.String),
        ("ListCost", pl.Decimal(18, 6)),
        ("ListUnitPrice", pl.Decimal(18, 6)),
        ("PricingCategory", pl.String),
        ("PricingQuantity", pl.Decimal(18, 6)),
        ("PricingUnit", pl.String),
        ("ProviderName", pl.String),
        ("PublisherName", pl.String),
        ("RegionId", pl.String),
        ("RegionName", pl.String),
        ("ResourceId", pl.String),
        ("ResourceName", pl.String),
        ("ResourceType", pl.String),
        ("ServiceCategory", pl.String),
        ("ServiceSubcategory", pl.String),
        ("ServiceName", pl.String),
        ("SubAccountId", pl.String),
        ("SubAccountName", pl.String),
        ("SubAccountType", pl.String),
        # Changed from pl.Object to pl.String to hold JSON metadata
        # (team_id, user_id, etc.) needed by Vantage Token Allocation.
        # This schema is only used for creating empty DataFrames (e.g.
        # when transform() receives no rows).  Parquet files are
        # self-describing and embed their own schema, so existing S3
        # exports are unaffected.  Previously Tags was always None.
        ("Tags", pl.String),
    ]
)

__all__ = ["FOCUS_NORMALIZED_SCHEMA"]
