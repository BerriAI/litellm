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
        ("Tags", pl.Object),
    ]
)

__all__ = ["FOCUS_NORMALIZED_SCHEMA"]
