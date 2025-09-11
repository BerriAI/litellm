# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, Optional

from pydantic import Field as FieldInfo

from ...._models import BaseModel

__all__ = ["GraderRunResponse", "Metadata", "MetadataErrors"]


class MetadataErrors(BaseModel):
    formula_parse_error: bool

    invalid_variable_error: bool

    api_model_grader_parse_error: bool = FieldInfo(alias="model_grader_parse_error")

    api_model_grader_refusal_error: bool = FieldInfo(alias="model_grader_refusal_error")

    api_model_grader_server_error: bool = FieldInfo(alias="model_grader_server_error")

    api_model_grader_server_error_details: Optional[str] = FieldInfo(
        alias="model_grader_server_error_details", default=None
    )

    other_error: bool

    python_grader_runtime_error: bool

    python_grader_runtime_error_details: Optional[str] = None

    python_grader_server_error: bool

    python_grader_server_error_type: Optional[str] = None

    sample_parse_error: bool

    truncated_observation_error: bool

    unresponsive_reward_error: bool


class Metadata(BaseModel):
    errors: MetadataErrors

    execution_time: float

    name: str

    sampled_model_name: Optional[str] = None

    scores: Dict[str, object]

    token_usage: Optional[int] = None

    type: str


class GraderRunResponse(BaseModel):
    metadata: Metadata

    api_model_grader_token_usage_per_model: Dict[str, object] = FieldInfo(
        alias="model_grader_token_usage_per_model"
    )

    reward: float

    sub_rewards: Dict[str, object]
