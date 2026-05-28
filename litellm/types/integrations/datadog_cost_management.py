from typing import Dict, List, Optional, TypedDict


from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class DatadogCostManagementInitParams(StandardCustomLoggerInitParams):
    """
    Init params for Datadog Cost Management
    """

    cost_tag_keys: Optional[List[str]] = None


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
    Tags: Optional[Dict[str, str]]
