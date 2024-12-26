from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

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


class PrometheusMetricLabels(Enum):
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


class UserAPIKeyLabelValues(BaseModel):
    end_user: Optional[str] = Field(None, alias=UserAPIKeyLabelNames.END_USER.value)
    user: Optional[str] = Field(None, alias=UserAPIKeyLabelNames.USER.value)
    hashed_api_key: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.API_KEY_HASH.value
    )
    api_key_alias: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.API_KEY_ALIAS.value
    )
    team: Optional[str] = Field(None, alias=UserAPIKeyLabelNames.TEAM.value)
    team_alias: Optional[str] = Field(None, alias=UserAPIKeyLabelNames.TEAM_ALIAS.value)
    requested_model: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.REQUESTED_MODEL.value
    )
    model: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value
    )
    litellm_model_name: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value
    )
    tags: List[str] = []
    model_id: Optional[str] = Field(None, alias=UserAPIKeyLabelNames.MODEL_ID.value)
    api_base: Optional[str] = Field(None, alias=UserAPIKeyLabelNames.API_BASE.value)
    api_provider: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.API_PROVIDER.value
    )
    exception_status: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.EXCEPTION_STATUS.value
    )
    exception_class: Optional[str] = Field(
        None, alias=UserAPIKeyLabelNames.EXCEPTION_CLASS.value
    )
