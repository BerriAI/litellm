"""Schema definitions for Focus export data."""

from __future__ import annotations

import polars as pl

# see: https://focus.finops.org/focus-specification/v1-2/
FOCUS_NORMALIZED_SCHEMA = pl.Schema(
    {
        "BilledCost": pl.Float64,
        "BillingAccountId": pl.String,
        "BillingAccountName": pl.String,
        "BillingCurrency": pl.String,
        "BillingPeriodStart": pl.Datetime(time_unit="us"),
        "BillingPeriodEnd": pl.Datetime(time_unit="us"),
        "ChargeCategory": pl.String,
        "ChargeClass": pl.String,
        "ChargeDescription": pl.String,
        "ChargeFrequency": pl.String,
        "ChargePeriodStart": pl.Datetime(time_unit="us"),
        "ChargePeriodEnd": pl.Datetime(time_unit="us"),
        "ConsumedQuantity": pl.Float64,
        "ConsumedUnit": pl.Float64,
        "ContractedCost": pl.Float64,
        "ContractedUnitPrice": pl.Float64,
        "EffectiveCost": pl.Float64,
        "InvoiceIssuerName": pl.String,
        "ListCost": pl.Float64,
        "ListUnitPrice": pl.Float64,
        "PricingCategory": pl.String,
        "PricingQuantity": pl.Float64,
        "PricingUnit": pl.String,
        "ProviderName": pl.String,
        "PublisherName": pl.String,
        "RegionId": pl.String,
        "RegionName": pl.String,
        "ResourceId": pl.String,
        "ResourceName": pl.String,
        "ResourceType": pl.String,
        "ServiceCategory": pl.String,
        "ServiceName": pl.String,
        "SubAccountId": pl.String,
        "SubAccountName": pl.String,
        "SubAccountType": pl.String,
        "Tags": pl.Object,
    }
)

__all__ = ["FOCUS_NORMALIZED_SCHEMA"]
