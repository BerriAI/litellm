from typing import TypedDict

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class DatadogCostManagementInitParams(StandardCustomLoggerInitParams):
    """
    Init params for Datadog Cost Management
    """

    cost_tag_keys: list[str] | None = None


class DatadogFOCUSCostEntry(TypedDict):
    """
    Represents a single cost line item in the FOCUS format.
    Ref: https://focus.finops.org/#specification
    """

    ProviderName: str
    ChargeDescription: str
    ChargePeriodStart: str
    ChargePeriodEnd: str
    BilledCost: float
    BillingCurrency: str
    Tags: dict[str, str] | None
