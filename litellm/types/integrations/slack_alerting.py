import os
from datetime import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict

from pydantic import BaseModel, Field

from litellm.types.utils import LiteLLMPydanticObjectBase

SLACK_ALERTING_THRESHOLD_5_PERCENT = 0.05
SLACK_ALERTING_THRESHOLD_15_PERCENT = 0.15


class BaseOutageModel(TypedDict):
    alerts: List[int]
    minor_alert_sent: bool
    major_alert_sent: bool
    last_updated_at: float


class OutageModel(BaseOutageModel):
    model_id: str


class ProviderRegionOutageModel(BaseOutageModel):
    provider_region_id: str
    deployment_ids: Set[str]


# we use this for the email header, please send a test email if you change this. verify it looks good on email
LITELLM_LOGO_URL = "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
LITELLM_SUPPORT_CONTACT = "support@berri.ai"


class SlackAlertingArgsEnum(Enum):
    daily_report_frequency = 12 * 60 * 60
    report_check_interval = 5 * 60
    budget_alert_ttl = 24 * 60 * 60
    outage_alert_ttl = 1 * 60
    region_outage_alert_ttl = 1 * 60
    minor_outage_alert_threshold = 1 * 5
    major_outage_alert_threshold = 1 * 10
    max_outage_alert_list_size = 1 * 10


class SlackAlertingArgs(LiteLLMPydanticObjectBase):
    daily_report_frequency: int = Field(
        default=int(
            os.getenv(
                "SLACK_DAILY_REPORT_FREQUENCY",
                int(SlackAlertingArgsEnum.daily_report_frequency.value),
            )
        ),
        description="Frequency of receiving deployment latency/failure reports. Default is 12hours. Value is in seconds.",
    )
    report_check_interval: int = Field(
        default=SlackAlertingArgsEnum.report_check_interval.value,
        description="Frequency of checking cache if report should be sent. Background process. Default is once per hour. Value is in seconds.",
    )  # 5 minutes
    budget_alert_ttl: int = Field(
        default=SlackAlertingArgsEnum.budget_alert_ttl.value,
        description="Cache ttl for budgets alerts. Prevents spamming same alert, each time budget is crossed. Value is in seconds.",
    )  # 24 hours
    outage_alert_ttl: int = Field(
        default=SlackAlertingArgsEnum.outage_alert_ttl.value,
        description="Cache ttl for model outage alerts. Sets time-window for errors. Default is 1 minute. Value is in seconds.",
    )  # 1 minute ttl
    region_outage_alert_ttl: int = Field(
        default=SlackAlertingArgsEnum.region_outage_alert_ttl.value,
        description="Cache ttl for provider-region based outage alerts. Alert sent if 2+ models in same region report errors. Sets time-window for errors. Default is 1 minute. Value is in seconds.",
    )  # 1 minute ttl
    minor_outage_alert_threshold: int = Field(
        default=SlackAlertingArgsEnum.minor_outage_alert_threshold.value,
        description="The number of errors that count as a model/region minor outage. ('400' error code is not counted).",
    )
    major_outage_alert_threshold: int = Field(
        default=SlackAlertingArgsEnum.major_outage_alert_threshold.value,
        description="The number of errors that countas a model/region major outage. ('400' error code is not counted).",
    )
    max_outage_alert_list_size: int = Field(
        default=SlackAlertingArgsEnum.max_outage_alert_list_size.value,
        description="Maximum number of errors to store in cache. For a given model/region. Prevents memory leaks.",
    )  # prevent memory leak
    log_to_console: bool = Field(
        default=False,
        description="If true, the alerting payload will be printed to the console.",
    )


class DeploymentMetrics(LiteLLMPydanticObjectBase):
    """
    Metrics per deployment, stored in cache

    Used for daily reporting
    """

    id: str
    """id of deployment in router model list"""

    failed_request: bool
    """did it fail the request?"""

    latency_per_output_token: Optional[float]
    """latency/output token of deployment"""

    updated_at: dt
    """Current time of deployment being updated"""


class SlackAlertingCacheKeys(Enum):
    """
    Enum for deployment daily metrics keys - {deployment_id}:{enum}
    """

    failed_requests_key = "failed_requests_daily_metrics"
    latency_key = "latency_daily_metrics"
    report_sent_key = "daily_metrics_report_sent"


class AlertType(str, Enum):
    """
    Enum for alert types and management event types
    """

    # LLM-related alerts
    llm_exceptions = "llm_exceptions"
    llm_too_slow = "llm_too_slow"
    llm_requests_hanging = "llm_requests_hanging"

    # Budget and spend alerts
    budget_alerts = "budget_alerts"
    spend_reports = "spend_reports"
    failed_tracking_spend = "failed_tracking_spend"

    # Database alerts
    db_exceptions = "db_exceptions"

    # Report alerts
    daily_reports = "daily_reports"

    # Deployment alerts
    cooldown_deployment = "cooldown_deployment"
    new_model_added = "new_model_added"

    # Outage alerts
    outage_alerts = "outage_alerts"
    region_outage_alerts = "region_outage_alerts"

    # Fallback alerts
    fallback_reports = "fallback_reports"

    # Virtual Key Events
    new_virtual_key_created = "new_virtual_key_created"
    virtual_key_updated = "virtual_key_updated"
    virtual_key_deleted = "virtual_key_deleted"

    # Team Events
    new_team_created = "new_team_created"
    team_updated = "team_updated"
    team_deleted = "team_deleted"

    # Internal User Events
    new_internal_user_created = "new_internal_user_created"
    internal_user_updated = "internal_user_updated"
    internal_user_deleted = "internal_user_deleted"


DEFAULT_ALERT_TYPES: List[AlertType] = [
    # LLM related alerts
    AlertType.llm_exceptions,
    AlertType.llm_too_slow,
    AlertType.llm_requests_hanging,
    # Budget and spend alerts
    AlertType.budget_alerts,
    AlertType.spend_reports,
    AlertType.failed_tracking_spend,
    # Database alerts
    AlertType.db_exceptions,
    # Report alerts
    AlertType.daily_reports,
    # Deployment alerts
    AlertType.cooldown_deployment,
    AlertType.new_model_added,
    # Outage alerts
    AlertType.outage_alerts,
    AlertType.region_outage_alerts,
    # Fallback alerts
    AlertType.fallback_reports,
]
