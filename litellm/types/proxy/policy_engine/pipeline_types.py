"""
Pipeline type definitions for guardrail pipelines.

Pipelines define ordered, conditional execution of guardrails within a policy.
When a policy has a `pipeline`, its guardrails run in the defined step order
with configurable actions on pass/fail, rather than independently.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_PIPELINE_ACTIONS = {"allow", "block", "next", "modify_response"}
VALID_PIPELINE_MODES = {"pre_call", "post_call"}


class PipelineStep(BaseModel):
    """
    A single step in a guardrail pipeline.

    Each step runs a guardrail and takes an action based on pass/fail.
    """

    guardrail: str = Field(description="Name of the guardrail to run.")
    on_fail: str = Field(
        default="block",
        description="Action when guardrail rejects: next | block | allow | modify_response",
    )
    on_pass: str = Field(
        default="allow",
        description="Action when guardrail passes: next | block | allow | modify_response",
    )
    pass_data: bool = Field(
        default=False,
        description="Forward modified request data (e.g., PII-masked) to next step.",
    )
    modify_response_message: Optional[str] = Field(
        default=None,
        description="Custom message for modify_response action.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("on_fail", "on_pass")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in VALID_PIPELINE_ACTIONS:
            raise ValueError(
                f"Invalid action '{v}'. Must be one of: {sorted(VALID_PIPELINE_ACTIONS)}"
            )
        return v


class GuardrailPipeline(BaseModel):
    """
    Defines ordered execution of guardrails with conditional actions.

    When present on a policy, the guardrails in `steps` are executed
    sequentially instead of independently.
    """

    mode: str = Field(description="Event hook: pre_call | post_call")
    steps: List[PipelineStep] = Field(
        description="Ordered list of pipeline steps. Must have at least 1 step.",
        min_length=1,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in VALID_PIPELINE_MODES:
            raise ValueError(
                f"Invalid mode '{v}'. Must be one of: {sorted(VALID_PIPELINE_MODES)}"
            )
        return v


class PipelineStepResult(BaseModel):
    """Result of executing a single pipeline step."""

    guardrail_name: str
    outcome: Literal["pass", "fail", "error"]
    action_taken: str
    modified_data: Optional[Dict[str, Any]] = None
    error_detail: Optional[str] = None
    duration_seconds: Optional[float] = None


class PipelineExecutionResult(BaseModel):
    """Result of executing an entire pipeline."""

    terminal_action: str  # block | allow | modify_response
    step_results: List[PipelineStepResult]
    modified_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    modify_response_message: Optional[str] = None
