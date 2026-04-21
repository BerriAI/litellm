import re
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

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


# v1: single translate pass + escape loop (avoids chained str.replace allocations).
_PROMETHEUS_LABEL_VALUE_TRANSLATE_V1 = str.maketrans("\n", " ", "\r\u2028\u2029")


def _sanitize_prometheus_label_value(value: Optional[Any]) -> Optional[str]:
    """
    Same semantics as :func:`_sanitize_prometheus_label_value`, implemented with
    ``str.translate`` plus a single escape pass instead of chained ``replace``.
    """
    if value is None:
        return None

    str_value: str = value if isinstance(value, str) else str(value)

    cleaned = str_value.translate(_PROMETHEUS_LABEL_VALUE_TRANSLATE_V1)
    if "\\" not in cleaned and '"' not in cleaned:
        return cleaned

    parts: List[str] = []
    append = parts.append
    for ch in cleaned:
        if ch == "\\":
            append("\\\\")
        elif ch == '"':
            append('\\"')
        else:
            append(ch)
    return "".join(parts)


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
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    420.0,  # 7 minutes
    600.0,  # 10 minutes (typical default LLM request timeout)
    float("inf"),
)

# Batch jobs can run for minutes to hours; buckets span 1 min → 24 h.
BATCH_DURATION_BUCKETS = (
    60.0,
    120.0,
    300.0,
    600.0,
    900.0,
    1800.0,
    3600.0,
    7200.0,
    14400.0,
    28800.0,
    43200.0,
    86400.0,
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
    STREAM = "stream"
    ORG_ID = "org_id"
    ORG_ALIAS = "org_alias"


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
    "litellm_remaining_org_budget_metric",
    "litellm_org_max_budget_metric",
    "litellm_org_budget_remaining_hours_metric",
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
    "litellm_in_flight_requests",
    # Managed batch metrics
    "litellm_managed_batch_created_total",
    "litellm_managed_file_size_bytes",
    "litellm_managed_batch_duration_seconds",
    "litellm_managed_file_created_total",
    "litellm_managed_file_deleted_total",
    "litellm_check_batch_cost_jobs_polled",
    "litellm_check_batch_cost_jobs_processed_total",
    "litellm_check_batch_cost_errors_total",
    "litellm_check_batch_cost_last_run_timestamp",
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
        UserAPIKeyLabelNames.API_PROVIDER.value,
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
        UserAPIKeyLabelNames.API_PROVIDER.value,
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

    litellm_remaining_org_budget_metric = [
        UserAPIKeyLabelNames.ORG_ID.value,
        UserAPIKeyLabelNames.ORG_ALIAS.value,
    ]

    litellm_org_max_budget_metric = [
        UserAPIKeyLabelNames.ORG_ID.value,
        UserAPIKeyLabelNames.ORG_ALIAS.value,
    ]

    litellm_org_budget_remaining_hours_metric = [
        UserAPIKeyLabelNames.ORG_ID.value,
        UserAPIKeyLabelNames.ORG_ALIAS.value,
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

    # Metrics whose emission paths supply org context (used by get_labels)
    _org_label_metrics: ClassVar[frozenset] = frozenset(
        {
            "litellm_llm_api_latency_metric",
            "litellm_llm_api_time_to_first_token_metric",
            "litellm_request_total_latency_metric",
            "litellm_request_queue_time_seconds",
            "litellm_proxy_total_requests_metric",
            "litellm_proxy_failed_requests_metric",
            "litellm_deployment_latency_per_output_token",
            "litellm_requests_metric",
            "litellm_spend_metric",
            "litellm_input_tokens_metric",
            "litellm_total_tokens_metric",
            "litellm_output_tokens_metric",
        }
    )

    # Managed batch metrics
    _batch_user_labels = [
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.USER_EMAIL.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
    ]

    litellm_managed_batch_created_total = _batch_user_labels

    litellm_managed_file_size_bytes: List[str] = (
        []
    )  # labels: purpose, file_type, model, api_provider, user (custom)

    litellm_managed_batch_duration_seconds = [
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
    ]

    litellm_managed_file_created_total = _batch_user_labels

    litellm_managed_file_deleted_total: List[str] = (
        []
    )  # only "result" label, added at metric creation

    litellm_check_batch_cost_jobs_polled: List[str] = []

    litellm_check_batch_cost_jobs_processed_total = [
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.API_PROVIDER.value,
    ]

    litellm_check_batch_cost_errors_total: List[str] = []  # label: error_type (custom)

    litellm_check_batch_cost_last_run_timestamp: List[str] = []

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

        # Conditionally add stream label to litellm_proxy_total_requests_metric
        if (
            label_name == "litellm_proxy_total_requests_metric"
            and litellm.prometheus_emit_stream_label is True
            and UserAPIKeyLabelNames.STREAM.value not in default_labels
        ):
            custom_labels.append(UserAPIKeyLabelNames.STREAM.value)

        if label_name in PrometheusMetricLabels._org_label_metrics:
            for label in [
                UserAPIKeyLabelNames.ORG_ID.value,
                UserAPIKeyLabelNames.ORG_ALIAS.value,
            ]:
                if label not in default_labels and label not in custom_labels:
                    custom_labels.append(label)

        return default_labels + custom_labels


_USER_API_KEY_LABEL_VALUE_INIT_ALIASES: Dict[str, str] = {
    # Some tests / call sites use ``api_key_hash``; Prometheus field is ``hashed_api_key``.
    "api_key_hash": "hashed_api_key",
}


@dataclass(frozen=True, init=False)
class UserAPIKeyLabelValues:
    """
    Prometheus metric label inputs (Python field names match historical Pydantic ``model_dump`` keys).

    Immutable value object: use ``dataclasses.replace()`` to derive a new instance.
    ``model_dump()`` is provided for call sites that still expect a Pydantic-like dict.
    """

    end_user: Optional[str] = None
    user: Optional[str] = None
    user_email: Optional[str] = None
    hashed_api_key: Optional[str] = None
    api_key_alias: Optional[str] = None
    team: Optional[str] = None
    team_alias: Optional[str] = None
    model_group: Optional[str] = None
    requested_model: Optional[str] = None
    model: Optional[str] = None
    litellm_model_name: Optional[str] = None
    # Accept list/tuple at construction time; normalize to tuple in __post_init__.
    tags: Union[Tuple[str, ...], List[str]] = ()
    custom_metadata_labels: Mapping[str, str] = field(default_factory=dict)
    model_id: Optional[str] = None
    api_base: Optional[str] = None
    api_provider: Optional[str] = None
    exception_status: Optional[str] = None
    exception_class: Optional[str] = None
    status_code: Optional[str] = None
    fallback_model: Optional[str] = None
    route: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    stream: Optional[str] = None
    org_id: Optional[str] = None
    org_alias: Optional[str] = None

    # Added for test compatibility.
    def __init__(self, **kwargs: Any) -> None:
        """
        Match former Pydantic behavior: unknown keys are ignored; ``api_key_hash`` maps to
        ``hashed_api_key``. This supports ``**standard_logging_payload`` in tests.
        """
        field_names = {f.name for f in fields(self)}
        merged: Dict[str, Any] = {}
        for f in fields(self):
            if f.default_factory is not MISSING:
                merged[f.name] = f.default_factory()
            else:
                merged[f.name] = f.default

        for k, v in kwargs.items():
            if k in field_names:
                merged[k] = v
                continue
            canon = _USER_API_KEY_LABEL_VALUE_INIT_ALIASES.get(k)
            if canon is not None and canon in field_names:
                merged[canon] = v

        for f in fields(self):
            object.__setattr__(self, f.name, merged[f.name])
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        if self.stream is not None:
            object.__setattr__(self, "stream", str(self.stream))
        _cmd = dict(self.custom_metadata_labels)
        object.__setattr__(
            self,
            "custom_metadata_labels",
            MappingProxyType(_cmd),
        )

    def __repr__(self) -> str:
        # Perf: this object is constructed on every Prometheus logging path; verbose
        # dataclass/Pydantic-style repr is expensive and often pulled in accidentally
        # via f-strings / debug logging. Return empty so accidental stringification
        # stays cheap. (Dataclass default `str()` delegates to `__repr__`.)
        return ""

    def model_dump(self) -> Dict[str, Any]:
        """Same shape as the former Pydantic ``model_dump()`` (plain dict, list tags)."""
        d: Dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        d["tags"] = list(self.tags)
        d["custom_metadata_labels"] = dict(self.custom_metadata_labels)
        return d


@dataclass
class PrometheusMetricsConfig:
    """Configuration for filtering Prometheus metrics (parsed once from proxy config)."""

    group: str
    metrics: List[str]
    include_labels: Optional[List[str]] = None


@dataclass
class PrometheusSettings:
    """Settings for Prometheus metrics configuration."""

    prometheus_metrics_config: Optional[List[PrometheusMetricsConfig]] = None


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
