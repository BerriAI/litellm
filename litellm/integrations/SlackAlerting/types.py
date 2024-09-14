import os
from datetime import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict

from pydantic import BaseModel, Field


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


class LiteLLMBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class SlackAlertingArgsEnum(Enum):
    daily_report_frequency: int = 12 * 60 * 60
    report_check_interval: int = 5 * 60
    budget_alert_ttl: int = 24 * 60 * 60
    outage_alert_ttl: int = 1 * 60
    region_outage_alert_ttl: int = 1 * 60
    minor_outage_alert_threshold: int = 1 * 5
    major_outage_alert_threshold: int = 1 * 10
    max_outage_alert_list_size: int = 1 * 10


class SlackAlertingArgs(LiteLLMBase):
    daily_report_frequency: int = Field(
        default=int(
            os.getenv(
                "SLACK_DAILY_REPORT_FREQUENCY",
                SlackAlertingArgsEnum.daily_report_frequency.value,
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


class DeploymentMetrics(LiteLLMBase):
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
