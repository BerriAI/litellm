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


class DeciderBranchCondition(BaseModel):
    """
    Condition for a decider branch: compare detection_info[key] with value using op.
    """

    key: str = Field(description="Key in detection_info to evaluate.")
    op: Literal["eq", "ne", "in"] = Field(
        description="Comparison: eq (==), ne (!=), in (val in value list)."
    )
    value: Any = Field(description="Value to compare against (list for 'in' op).")

    model_config = ConfigDict(extra="forbid")


class DeciderBranch(BaseModel):
    """
    Branch target for decider steps. condition=None is the default branch.
    """

    condition: Optional[DeciderBranchCondition] = Field(
        default=None,
        description="When None, this is the default branch; otherwise branch when condition matches.",
    )
    next_step_index: int = Field(
        description="Index of the next pipeline step to run when this branch is taken.",
        ge=0,
    )

    model_config = ConfigDict(extra="forbid")


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
    decider_branches: Optional[List[DeciderBranch]] = Field(
        default=None,
        description="When set and step passes, branch to next_step_index based on detection_info; condition None = default branch.",
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
