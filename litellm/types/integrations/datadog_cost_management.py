from typing import Dict, Optional, TypedDict


from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class DatadogCostManagementInitParams(StandardCustomLoggerInitParams):
    """
    Init params for Datadog Cost Management
    """

    datadog_cost_management_params: Optional[Dict] = None


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
