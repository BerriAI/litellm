"""Schema definitions for Focus export data."""

from __future__ import annotations

from typing import Final

import polars as pl

FOCUS_NORMALIZED_SCHEMA: Final = pl.Schema(
    [
        ("BilledCost", pl.Decimal(18, 6)),
        ("BillingAccountId", pl.String),
        ("BillingAccountName", pl.String),
        ("BillingAccountType", pl.String),
        ("BillingCurrency", pl.String),
        ("BillingPeriodStart", pl.String),
        ("BillingPeriodEnd", pl.String),
        ("ChargeCategory", pl.String),
        ("ChargeClass", pl.String),
        ("ChargeDescription", pl.String),
        ("ChargeFrequency", pl.String),
        ("ChargePeriodStart", pl.String),
        ("ChargePeriodEnd", pl.String),
        ("ConsumedQuantity", pl.Decimal(18, 6)),
        ("ConsumedUnit", pl.String),
        ("ContractedCost", pl.Decimal(18, 6)),
        ("ContractedUnitPrice", pl.Decimal(18, 6)),
        ("CredentialId", pl.String),
        ("EffectiveCost", pl.Decimal(18, 6)),
        ("HostProviderName", pl.String),
        ("InvoiceId", pl.String),
        ("InvoiceIssuerName", pl.String),
        ("ListCost", pl.Decimal(18, 6)),
        ("ListUnitPrice", pl.Decimal(18, 6)),
        ("AvailabilityZone", pl.String),
        ("PricingCategory", pl.String),
        ("PricingCurrency", pl.String),
        ("PricingCurrencyContractedUnitPrice", pl.Decimal(18, 6)),
        ("PricingCurrencyEffectiveCost", pl.Decimal(18, 6)),
        ("PricingCurrencyListUnitPrice", pl.Decimal(18, 6)),
        ("PricingQuantity", pl.Decimal(18, 6)),
        ("PricingUnit", pl.String),
        ("PrincipalId", pl.String),
        ("RegionId", pl.String),
        ("RegionName", pl.String),
        ("RequesterDetails", pl.String),
        ("ResourceId", pl.String),
        ("ResourceName", pl.String),
        ("ResourceType", pl.String),
        ("ServiceCategory", pl.String),
        ("ServiceSubcategory", pl.String),
        ("ServiceName", pl.String),
        ("ServiceProviderName", pl.String),
        ("SkuId", pl.String),
        ("SkuPriceId", pl.String),
        ("SkuPriceDetails", pl.String),
        ("SubAccountId", pl.String),
        ("SubAccountName", pl.String),
        ("SubAccountType", pl.String),
        ("Tags", pl.String),
    ]
)

__all__ = ["FOCUS_NORMALIZED_SCHEMA"]
