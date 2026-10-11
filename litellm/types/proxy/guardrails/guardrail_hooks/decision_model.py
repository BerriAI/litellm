from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from litellm.types.decisions import MAX_DECISION_QUESTIONS
from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel


class DecisionModelCheck(LiteLLMBaseModel):
    """One predicate the decision model scores the request or response against."""

    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    action: Literal["block", "log"] = "block"

    @field_validator("instructions")
    @classmethod
    def _instructions_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value

    model_config = ConfigDict(frozen=True)


class DecisionModelGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    decision_model: str = Field(
        description="Decisions-API model that scores each check (e.g. a Jev deployment on this proxy)"
    )
    checks: tuple[DecisionModelCheck, ...] = Field(
        min_length=1,
        max_length=MAX_DECISION_QUESTIONS,
        description=(
            "Predicates the decision model answers about each request or response. A check flagged at or "
            "above its threshold blocks (action 'block') or is recorded (action 'log')."
        ),
    )
    max_input_chars: int = Field(
        default=24000,
        gt=0,
        description="Character budget for each text sent to the decision model. Texts longer than this are split into overlapping chunks that are each checked, so long inputs are fully screened",
    )
    max_concurrent_decision_calls: int = Field(
        default=8,
        gt=0,
        description="Maximum decisions calls in flight at once on this guardrail in each proxy worker. Calls beyond it wait for a free slot",
    )
    streaming_end_of_stream_only: bool = Field(
        default=False,
        description="For streamed responses, skip the scan every streaming_sampling_rate chunks and score the assembled response once at end of stream",
    )
    streaming_sampling_rate: int = Field(
        default=5,
        ge=1,
        description="For streamed responses, score the accumulated response every Nth chunk. Each scan is a decisions call",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Decision Model"

    model_config = ConfigDict(frozen=True)
