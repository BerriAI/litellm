"""
FOCUS (FinOps Open Cost & Usage Specification) types for LiteLLM integration.

FOCUS is an open specification for consistent cost and usage datasets from the FinOps Foundation.
More info: https://focus.finops.org/
"""

from typing import Any, Dict, Optional


class FOCUSRecord(Dict[str, Any]):
    """FOCUS (FinOps Open Cost & Usage Specification) record structure.
    
    This class represents a FOCUS record created from LiteLLM usage data.
    FOCUS is an open standard for cloud cost and usage data interoperability.
    
    Required FOCUS columns:
    - BilledCost: The cost that is invoiced/billed (float)
    - BillingPeriodStart: Start of billing period (ISO datetime string)
    - BillingPeriodEnd: End of billing period (ISO datetime string)
    
    Recommended FOCUS columns for LLM usage:
    - ChargeCategory: Type of charge (e.g., "Usage")
    - ChargeClass: Classification of charge (e.g., "Standard")
    - ChargeDescription: Human-readable description
    - ChargePeriodStart: Start of charge period (ISO datetime string)
    - ChargePeriodEnd: End of charge period (ISO datetime string)
    - ConsumedQuantity: Amount of resource consumed (tokens)
    - ConsumedUnit: Unit of consumption (e.g., "Tokens")
    - EffectiveCost: Cost after any applicable discounts
    - ListCost: Cost at list/published prices
    - ProviderName: Name of the LLM provider (e.g., "OpenAI", "Anthropic")
    - PublisherName: Publisher/vendor name (e.g., "LiteLLM")
    - Region: Geographic region (if applicable)
    - ResourceId: Unique identifier for the resource
    - ResourceName: Human-readable resource name (model name)
    - ResourceType: Type of resource (e.g., "LLM")
    - ServiceCategory: Category of service (e.g., "AI and Machine Learning")
    - ServiceName: Name of service (e.g., "LLM Inference")
    - SubAccountId: Sub-account identifier (team_id)
    - SubAccountName: Sub-account name (team_alias)
    - Tags: Additional metadata tags
    """
    pass


# Type alias for better readability in function signatures
FOCUSRecordDict = Dict[str, Any]


class FOCUSExportConfig:
    """Configuration for FOCUS data export."""
    
    def __init__(
        self,
        include_tags: bool = True,
        include_token_breakdown: bool = True,
        timezone: str = "UTC",
    ):
        """
        Initialize FOCUS export configuration.
        
        Args:
            include_tags: Whether to include resource tags in export
            include_token_breakdown: Whether to include prompt/completion token breakdown
            timezone: Timezone for date handling (default: UTC)
        """
        self.include_tags = include_tags
        self.include_token_breakdown = include_token_breakdown
        self.timezone = timezone
