"""Configuration and classifier output for capability routing."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self


def _nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _provider_model(value: str, field_name: str) -> str:
    normalized = _nonblank(value, field_name)
    if normalized.startswith("auto_router/"):
        raise ValueError(f"{field_name} must name a provider model group")
    return normalized


class CapabilityRouterCandidate(BaseModel):
    """A model group and the operator's description of when it succeeds."""

    model: str
    description: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _provider_model(value, "candidate model")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _nonblank(value, "candidate description")


class CapabilityClassifierConfig(BaseModel):
    """The model used to estimate each candidate's probability of success."""

    model: str
    timeout_ms: int = Field(default=3000, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _provider_model(value, "classifier model")


class CapabilityRouterConfig(BaseModel):
    """Operator configuration for cheapest-qualified model selection."""

    candidates: tuple[CapabilityRouterCandidate, ...] = Field(min_length=2)
    classifier: CapabilityClassifierConfig
    probability_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    fallback_model: str
    estimated_output_tokens: int = Field(default=1000, ge=1)
    cache_ttl_seconds: int = Field(default=3600, ge=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("fallback_model")
    @classmethod
    def validate_fallback_model(cls, value: str) -> str:
        return _provider_model(value, "fallback_model")

    @field_validator("probability_threshold", mode="before")
    @classmethod
    def validate_finite_threshold(cls, value: object) -> object:
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("probability_threshold must be finite")
        return value

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        models = tuple(candidate.model for candidate in self.candidates)
        if len(models) != len(set(models)):
            raise ValueError("candidate model names must be unique")
        if self.fallback_model not in models:
            raise ValueError("fallback_model must be one of the candidate models")
        return self


class CapabilityCandidateScore(BaseModel):
    """One classifier estimate for the current task."""

    model: str
    p_solve: float = Field(ge=0.0, le=1.0)
    reason: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("model", "reason")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("p_solve", mode="before")
    @classmethod
    def validate_probability(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("p_solve must be a JSON number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("p_solve must be finite")
        return value


class CapabilityClassifierVerdict(BaseModel):
    """Strict structured output returned by the classifier."""

    candidates: tuple[CapabilityCandidateScore, ...] = Field(min_length=2)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_unique_models(self) -> Self:
        models = tuple(candidate.model for candidate in self.candidates)
        if len(models) != len(set(models)):
            raise ValueError("classifier candidate model names must be unique")
        return self


CapabilitySelectionReason = Literal[
    "cheapest_qualified",
    "no_qualified_candidate",
    "missing_candidate_price",
    "classifier_error",
    "invalid_classifier_verdict",
]
