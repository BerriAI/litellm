import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from typing_extensions import Annotated

import litellm


def _sanitize_prometheus_label_name(label: str) -> str:
    """
    Sanitize a label name to comply with Prometheus label name requirements.

    Prometheus label names must match: ^[a-zA-Z_][a-zA-Z0-9_]*$
    - First character: letter (a-z, A-Z) or underscore (_)
    - Subsequent characters: letters, digits (0-9), or underscores (_)

    Args:
        label: The label name to sanitize

    Returns:
        A sanitized label name that complies with Prometheus requirements
    """
    if not label:
        return "_"

    # Replace all invalid characters with underscores
    # Keep only letters, digits, and underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", label)

    # Ensure first character is valid (letter or underscore)
    if sanitized and not re.match(r"^[a-zA-Z_]", sanitized[0]):
        sanitized = "_" + sanitized

    # Handle empty string after sanitization
    if not sanitized:
        sanitized = "_"

    return sanitized


def _sanitize_prometheus_label_value(value: Optional[Any]) -> Optional[str]:
    """
    Sanitize a label value for Prometheus text format compatibility.

    Removes or replaces characters that break the Prometheus exposition format:
    - U+2028 (Line Separator) and U+2029 (Paragraph Separator) are removed
    - Carriage returns are removed
    - Newlines are replaced with spaces
    - Backslashes and double quotes are escaped per Prometheus spec
    """
    if value is None:
        return None

    # Coerce non-string values (int, bool, etc.) to str before sanitizing
    if not isinstance(value, str):
        value = str(value)

    # Remove Unicode line/paragraph separators that break text format
    value = value.replace("\u2028", "").replace("\u2029", "")

    # Remove carriage returns
    value = value.replace("\r", "")

    # Replace newlines with spaces
    value = value.replace("\n", " ")

    # Escape backslashes and double quotes per Prometheus exposition format
    value = value.replace("\\", "\\\\").replace('"', '\\"')

    return value


@dataclass
class MetricValidationError:
    """Error for invalid metric name"""

    metric_name: str
    valid_metrics: Tuple[str, ...]

    @property
    def message(self) -> str:
        return f"Invalid metric name: {self.metric_name}"


@dataclass
class LabelValidationError:
    """Error for invalid labels on a metric"""

    metric_name: str
    invalid_labels: List[str]
    valid_labels: List[str]

    @property
    def message(self) -> str:
        return f"Invalid labels for metric '{self.metric_name}': {self.invalid_labels}"


@dataclass
class ValidationResults:
    """Container for all validation results"""

    metric_errors: List[MetricValidationError]
    label_errors: List[LabelValidationError]

    @property
    def has_errors(self) -> bool:
        return bool(self.metric_errors or self.label_errors)

    @property
    def all_error_messages(self) -> List[str]:
        messages = [error.message for error in self.metric_errors]
        messages.extend([error.message for error in self.label_errors])
        return messages


REQUESTED_MODEL = "requested_model"
EXCEPTION_STATUS = "exception_status"
EXCEPTION_CLASS = "exception_class"
STATUS_CODE = "status_code"
EXCEPTION_LABELS = [EXCEPTION_STATUS, EXCEPTION_CLASS]
LATENCY_BUCKETS = (
    0.005,
    0.00625,
    0.0125,
    0.025,
    0.05,
    0.1,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    60.0,
    120.0,
    180.0,
    240.0,
    300.0,
    float("inf"),
)


class UserAPIKeyLabelNames(Enum):
    END_USER = "end_user"
    USER = "user"
    USER_EMAIL = "user_email"
    API_KEY_HASH = "hashed_api_key"
    API_KEY_ALIAS = "api_key_alias"
    TEAM = "team"
    TEAM_ALIAS = "team_alias"
    REQUESTED_MODEL = REQUESTED_MODEL
    v1_LITELLM_MODEL_NAME = "model"
    v2_LITELLM_MODEL_NAME = "litellm_model_name"
    TAG = "tag"
    MODEL_ID = "model_id"
    API_BASE = "api_base"
    API_PROVIDER = "api_provider"
    EXCEPTION_STATUS = EXCEPTION_STATUS
    EXCEPTION_CLASS = EXCEPTION_CLASS
    STATUS_CODE = "status_code"
    FALLBACK_MODEL = "fallback_model"
    ROUTE = "route"
    MODEL_GROUP = "model_group"
    CLIENT_IP = "client_ip"
    USER_AGENT = "user_agent"
    CALLBACK_NAME = "callback_name"


DEFINED_PROMETHEUS_METRICS = Literal[
    "litellm_llm_api_latency_metric",
    "litellm_llm_api_time_to_first_token_metric",
    "litellm_request_total_latency_metric",
    "litellm_overhead_latency_metric",
    "litellm_remaining_requests_metric",
    "litellm_remaining_tokens_metric",
    "litellm_proxy_total_requests_metric",
    "litellm_proxy_failed_requests_metric",
    "litellm_deployment_latency_per_output_token",
    "litellm_requests_metric",
    "litellm_spend_metric",
    "litellm_total_tokens_metric",
    "litellm_input_tokens_metric",
    "litellm_output_tokens_metric",
    "litellm_deployment_successful_fallbacks",
    "litellm_deployment_failed_fallbacks",
    "litellm_remaining_team_budget_metric",
    "litellm_team_max_budget_metric",
    "litellm_team_budget_remaining_hours_metric",
    "litellm_remaining_api_key_budget_metric",
    "litellm_api_key_max_budget_metric",
    "litellm_api_key_budget_remaining_hours_metric",
    "litellm_remaining_user_budget_metric",
    "litellm_user_max_budget_metric",
    "litellm_user_budget_remaining_hours_metric",
    "litellm_deployment_state",
    "litellm_deployment_failure_responses",
    "litellm_deployment_total_requests",
    "litellm_deployment_success_responses",
    "litellm_deployment_cooled_down",
    "litellm_pod_lock_manager_size",
    "litellm_in_memory_daily_spend_update_queue_size",
    "litellm_redis_daily_spend_update_queue_size",
    "litellm_in_memory_spend_update_queue_size",
    "litellm_redis_spend_update_queue_size",
    "litellm_request_queue_time_seconds",
    "litellm_guardrail_latency_seconds",
    "litellm_guardrail_errors_total",
    "litellm_guardrail_requests_total",
    # Cache metrics
    "litellm_cache_hits_metric",
    "litellm_cache_misses_metric",
    "litellm_cached_tokens_metric",
    "litellm_deployment_tpm_limit",
    "litellm_deployment_rpm_limit",
    "litellm_remaining_api_key_requests_for_model",
    "litellm_remaining_api_key_tokens_for_model",
    "litellm_llm_api_failed_requests_metric",
    "litellm_callback_logging_failures_metric",
]


class PrometheusMetricLabels:
    litellm_llm_api_latency_metric = [
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_llm_api_time_to_first_token_metric = [
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_request_total_latency_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_request_queue_time_seconds = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    # Guardrail metrics - these use custom labels (guardrail_name, status, error_type, hook_type)
    # which are not part of UserAPIKeyLabelNames
    litellm_guardrail_latency_seconds: List[str] = []
    litellm_guardrail_errors_total: List[str] = []
    litellm_guardrail_requests_total: List[str] = []

    litellm_proxy_total_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.STATUS_CODE.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.ROUTE.value,
        UserAPIKeyLabelNames.CLIENT_IP.value,
        UserAPIKeyLabelNames.USER_AGENT.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_proxy_failed_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.EXCEPTION_STATUS.value,
        UserAPIKeyLabelNames.EXCEPTION_CLASS.value,
        UserAPIKeyLabelNames.ROUTE.value,
        UserAPIKeyLabelNames.CLIENT_IP.value,
        UserAPIKeyLabelNames.USER_AGENT.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_deployment_latency_per_output_token = [
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
    ]

    litellm_overhead_latency_metric = [
        UserAPIKeyLabelNames.MODEL_GROUP.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_remaining_requests_metric = [
        UserAPIKeyLabelNames.MODEL_GROUP.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_remaining_tokens_metric = [
        UserAPIKeyLabelNames.MODEL_GROUP.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.CLIENT_IP.value,
        UserAPIKeyLabelNames.USER_AGENT.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_spend_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.CLIENT_IP.value,
        UserAPIKeyLabelNames.USER_AGENT.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_input_tokens_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_total_tokens_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_output_tokens_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_deployment_state = [
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
    ]

    litellm_deployment_tpm_limit = [
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
    ]

    litellm_deployment_rpm_limit = litellm_deployment_tpm_limit

    litellm_deployment_cooled_down = [
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.EXCEPTION_STATUS.value,
    ]

    litellm_deployment_successful_fallbacks = [
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.FALLBACK_MODEL.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.EXCEPTION_STATUS.value,
        UserAPIKeyLabelNames.EXCEPTION_CLASS.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_deployment_failed_fallbacks = litellm_deployment_successful_fallbacks

    litellm_remaining_team_budget_metric = [
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
    ]

    litellm_team_max_budget_metric = [
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
    ]

    litellm_team_budget_remaining_hours_metric = [
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
    ]

    litellm_remaining_api_key_budget_metric = [
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
    ]

    litellm_api_key_max_budget_metric = litellm_remaining_api_key_budget_metric

    litellm_api_key_budget_remaining_hours_metric = (
        litellm_remaining_api_key_budget_metric
    )

    litellm_remaining_user_budget_metric = [
        UserAPIKeyLabelNames.USER.value,
    ]

    litellm_user_max_budget_metric = [
        UserAPIKeyLabelNames.USER.value,
    ]

    litellm_user_budget_remaining_hours_metric = [
        UserAPIKeyLabelNames.USER.value,
    ]

    litellm_user_budget_remaining_hours_metric = [
        UserAPIKeyLabelNames.USER.value,
    ]

    litellm_remaining_api_key_requests_for_model = [
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
    ]

    litellm_remaining_api_key_tokens_for_model = [
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
    ]

    litellm_callback_logging_failures_metric = [
        UserAPIKeyLabelNames.CALLBACK_NAME.value,
    ]

    # Add deployment metrics
    litellm_deployment_failure_responses = [
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.EXCEPTION_STATUS.value,
        UserAPIKeyLabelNames.EXCEPTION_CLASS.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.CLIENT_IP.value,
        UserAPIKeyLabelNames.USER_AGENT.value,
    ]

    litellm_deployment_total_requests = [
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
        UserAPIKeyLabelNames.API_BASE.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.CLIENT_IP.value,
        UserAPIKeyLabelNames.USER_AGENT.value,
    ]

    litellm_deployment_success_responses = litellm_deployment_total_requests

    litellm_remaining_api_key_requests_for_model = [
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_remaining_api_key_tokens_for_model = [
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_llm_api_failed_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    # Buffer monitoring metrics - these typically don't need additional labels
    litellm_pod_lock_manager_size: List[str] = []

    litellm_in_memory_daily_spend_update_queue_size: List[str] = []

    litellm_redis_daily_spend_update_queue_size: List[str] = []

    litellm_in_memory_spend_update_queue_size: List[str] = []

    litellm_redis_spend_update_queue_size: List[str] = []

    # Cache metrics - track cache hits, misses, and tokens served from cache
    _cache_metric_labels = [
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.MODEL_ID.value,
    ]

    litellm_cache_hits_metric = _cache_metric_labels
    litellm_cache_misses_metric = _cache_metric_labels
    litellm_cached_tokens_metric = _cache_metric_labels

    @staticmethod
    def get_labels(label_name: DEFINED_PROMETHEUS_METRICS) -> List[str]:
        default_labels = getattr(PrometheusMetricLabels, label_name)
        custom_labels = []

        # Add custom metadata labels
        custom_labels.extend(
            [
                _sanitize_prometheus_label_name(metric)
                for metric in litellm.custom_prometheus_metadata_labels
            ]
        )

        # Add custom tags labels
        custom_labels.extend(
            [
                _sanitize_prometheus_label_name(f"tag_{tag}")
                for tag in litellm.custom_prometheus_tags
            ]
        )

        return default_labels + custom_labels


class UserAPIKeyLabelValues(BaseModel):
    end_user: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.END_USER.value)
    ] = None
    user: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.USER.value)
    ] = None
    user_email: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.USER_EMAIL.value)
    ] = None
    hashed_api_key: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.API_KEY_HASH.value)
    ] = None
    api_key_alias: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.API_KEY_ALIAS.value)
    ] = None
    team: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.TEAM.value)
    ] = None
    team_alias: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.TEAM_ALIAS.value)
    ] = None
    model_group: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.MODEL_GROUP.value)
    ] = None
    requested_model: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.REQUESTED_MODEL.value)
    ] = None
    model: Annotated[
        Optional[str],
        Field(..., alias=UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value),
    ] = None
    litellm_model_name: Annotated[
        Optional[str],
        Field(..., alias=UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value),
    ] = None
    tags: List[str] = []
    custom_metadata_labels: Dict[str, str] = {}
    model_id: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.MODEL_ID.value)
    ] = None
    api_base: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.API_BASE.value)
    ] = None
    api_provider: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.API_PROVIDER.value)
    ] = None
    exception_status: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.EXCEPTION_STATUS.value)
    ] = None
    exception_class: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.EXCEPTION_CLASS.value)
    ] = None
    status_code: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.STATUS_CODE.value)
    ] = None
    fallback_model: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.FALLBACK_MODEL.value)
    ] = None
    route: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.ROUTE.value)
    ] = None
    client_ip: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.CLIENT_IP.value)
    ] = None
    user_agent: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.USER_AGENT.value)
    ] = None


class PrometheusMetricsConfig(BaseModel):
    """Configuration for filtering Prometheus metrics"""

    group: str = Field(..., description="Group name for this set of metrics")
    metrics: List[str] = Field(
        ..., description="List of metric names to include in this group"
    )
    include_labels: Optional[List[str]] = Field(
        None,
        description="List of labels to include for these metrics. If None, includes all default labels.",
    )


class PrometheusSettings(BaseModel):
    """Settings for Prometheus metrics configuration"""

    prometheus_metrics_config: Optional[List[PrometheusMetricsConfig]] = Field(
        None,
        description="Configuration for filtering Prometheus metrics by groups and labels",
    )


class NoOpMetric:
    """A no-op metric that has the same interface as prometheus metrics but does nothing"""

    def __init__(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass
