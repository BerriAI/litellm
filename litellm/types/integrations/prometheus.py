from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

import litellm

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


DEFINED_PROMETHEUS_METRICS = Literal[
    "litellm_llm_api_latency_metric",
    "litellm_request_total_latency_metric",
    "litellm_proxy_total_requests_metric",
    "litellm_proxy_failed_requests_metric",
    "litellm_deployment_latency_per_output_token",
    "litellm_requests_metric",
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
    ]

    litellm_proxy_total_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.STATUS_CODE.value,
    ]

    litellm_proxy_failed_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.EXCEPTION_STATUS.value,
        UserAPIKeyLabelNames.EXCEPTION_CLASS.value,
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

    litellm_requests_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
    ]

    litellm_input_tokens_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
    ]

    litellm_output_tokens_metric = [
        UserAPIKeyLabelNames.END_USER.value,
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.USER.value,
        UserAPIKeyLabelNames.REQUESTED_MODEL.value,
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

    @staticmethod
    def get_labels(label_name: DEFINED_PROMETHEUS_METRICS) -> List[str]:
        default_labels = getattr(PrometheusMetricLabels, label_name)
        return default_labels + [
            metric.replace(".", "_")
            for metric in litellm.custom_prometheus_metadata_labels
        ]


from typing import List, Optional

from pydantic import BaseModel, Field


class UserAPIKeyLabelValues(BaseModel):
    end_user: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.END_USER.value)
    ] = None
    user: Annotated[
        Optional[str], Field(..., alias=UserAPIKeyLabelNames.USER.value)
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
