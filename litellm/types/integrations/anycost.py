"""
AnyCost Integration Types for LiteLLM

Types for CloudZero AnyCost integration.
"""

import os
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnyCostChargeBy(str, Enum):
    """Enum for different charging methods"""
    TEAMS = "teams"
    USERS = "users"
    API_KEYS = "api_keys"
    END_USERS = "end_users"


class AnyCostConfig(BaseModel):
    """Configuration for AnyCost integration"""
    # CloudZero API Configuration
    api_key: str = Field(default_factory=lambda: os.getenv("CLOUDZERO_API_KEY", ""))
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "CLOUDZERO_BASE_URL",
            "https://api.cloudzero.com"
        )
    )

    # AWS S3 Configuration for CBF uploads
    s3_bucket: str = Field(
        default_factory=lambda: os.getenv("ANYCOST_S3_BUCKET", "")
    )
    s3_prefix: str = Field(
        default_factory=lambda: os.getenv("ANYCOST_S3_PREFIX", "litellm/cbf/")
    )

    # Integration Configuration
    charge_by: AnyCostChargeBy = Field(
        default_factory=lambda: AnyCostChargeBy(
            os.getenv("ANYCOST_CHARGE_BY", "teams")
        )
    )

    # CBF Configuration
    cbf_file_format: bool = Field(
        default_factory=lambda: os.getenv("ANYCOST_CBF_FORMAT", "true").lower() in [
            "true", "1", "yes"
        ]
    )
    telemetry_format: bool = Field(
        default_factory=lambda: os.getenv(
            "ANYCOST_TELEMETRY_FORMAT", "false"
        ).lower() in ["true", "1", "yes"]
    )

    # Scheduling
    send_daily_reports: bool = Field(
        default_factory=lambda: os.getenv(
            "ANYCOST_SEND_DAILY_REPORTS", "true"
        ).lower() in ["true", "1", "yes"]
    )


class CBFUsageInfo(BaseModel):
    """Usage information for CBF line item"""
    metric_name: str
    usage_amount: float
    usage_unit: str


class CBFLineItem(BaseModel):
    """Individual line item in CBF format"""
    # Required fields
    invoice_id: str
    billing_period_start_date: str
    billing_period_end_date: str
    line_item_id: str
    line_item_type: str = "Usage"

    # Product information
    product_name: str = "LiteLLM"
    service_name: str = "AI/ML"
    service_category: str = "Compute"

    # Usage information
    usage_info: List[CBFUsageInfo]

    # Resource information
    resource_id: str
    resource_name: Optional[str] = None
    resource_type: str = "API"

    # Cost information
    billing_currency: str = "USD"
    list_cost: float
    unblended_cost: float
    blended_cost: float

    # Tags for categorization
    tags: Dict[str, str] = Field(default_factory=dict)


class CBFBillingPeriod(BaseModel):
    """Billing period for CBF file"""
    start_date: str
    end_date: str
    currency: str = "USD"


class CBFFile(BaseModel):
    """Common Bill Format file structure"""
    version: str = "1.0"
    billing_period: CBFBillingPeriod
    line_items: List[CBFLineItem]


class TelemetryRecord(BaseModel):
    """Record for CloudZero Telemetry API"""
    timestamp: str
    source: str = "litellm"
    metric_name: str
    value: float
    unit: str = "USD"
    dimensions: Dict[str, str] = Field(default_factory=dict)


class AnyCostMetrics(BaseModel):
    """Aggregated metrics for AnyCost reporting"""
    spend: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Breakdown by provider and model
    providers: Dict[str, float] = Field(default_factory=dict)
    models: Dict[str, float] = Field(default_factory=dict)


class AnyCostChargeByEnum(str, Enum):
    """Charge by enum for configuration"""
    TEAMS = "teams"
    USERS = "users"
    API_KEYS = "api_keys"
    END_USERS = "end_users"
