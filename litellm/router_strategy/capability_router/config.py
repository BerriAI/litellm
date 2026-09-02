"""Configuration and classifier output for capability routing."""

import math
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self


def _nonblank(value: str, field_name: str) -> str:
    normalized: Final = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _provider_model(value: str, field_name: str) -> str:
    normalized: Final = _nonblank(value, field_name)
    if normalized.startswith("auto_router/"):
        raise ValueError(f"{field_name} must name a provider model group")
    return normalized


CapabilityRuleBoundary: TypeAlias = Literal["supported", "uncertain", "unsupported"]


class CapabilityRule(BaseModel):
    """One operator-declared condition and the coverage it implies when it matches the task."""

    boundary: CapabilityRuleBoundary
    rule: str
    observed_success_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("rule")
    @classmethod
    def validate_rule(cls, value: str) -> str:
        return _nonblank(value, "capability rule")


class CapabilityCalibrationBin(BaseModel):
    """One monotonic post-hoc calibration bucket learned from end-to-end outcomes."""

    upper_bound: float = Field(ge=0.0, le=1.0)
    probability: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("upper_bound", "probability", mode="before")
    @classmethod
    def validate_finite_value(cls, value: object) -> object:
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("calibration values must be finite")
        return value


class CapabilityRouterCandidate(BaseModel):
    """A model group and the operator's description of when it succeeds."""

    model: str
    description: str
    rules: tuple[CapabilityRule, ...] = ()
    probability_calibration: tuple[CapabilityCalibrationBin, ...] = ()

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _provider_model(value, "candidate model")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _nonblank(value, "candidate description")

    @model_validator(mode="after")
    def validate_probability_calibration(self) -> Self:
        upper_bounds: Final = tuple(bucket.upper_bound for bucket in self.probability_calibration)
        probabilities: Final = tuple(bucket.probability for bucket in self.probability_calibration)
        if any(right <= left for left, right in zip(upper_bounds, upper_bounds[1:])):
            raise ValueError("calibration upper bounds must be strictly increasing")
        if any(right < left for left, right in zip(probabilities, probabilities[1:])):
            raise ValueError("calibration probabilities must be nondecreasing")
        if upper_bounds and upper_bounds[-1] != 1.0:
            raise ValueError("the final calibration upper bound must be 1")
        return self


def indexed_rules(candidate: CapabilityRouterCandidate) -> tuple[tuple[str, CapabilityRule], ...]:
    """Pair each rule with the opaque id the prompt shows and the policy resolves."""
    return tuple((f"R{index + 1}", rule) for index, rule in enumerate(candidate.rules))


class CapabilityClassifierConfig(BaseModel):
    """The model used to estimate each candidate's probability of success."""

    model: str
    timeout_ms: int = Field(default=3000, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    max_message_chars: int = Field(default=2000, ge=1)

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
    threshold_step: float = Field(default=0.1, ge=0.0, le=1.0)
    fallback_model: str
    estimated_output_tokens: int = Field(default=1000, ge=1)
    cache_ttl_seconds: int = Field(default=3600, ge=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("fallback_model")
    @classmethod
    def validate_fallback_model(cls, value: str) -> str:
        return _provider_model(value, "fallback_model")

    @field_validator("probability_threshold", "threshold_step", mode="before")
    @classmethod
    def validate_finite_threshold(cls, value: object) -> object:
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("threshold values must be finite")
        return value

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        models: Final = tuple(candidate.model for candidate in self.candidates)
        if len(models) != len(frozenset(models)):
            raise ValueError("candidate model names must be unique")
        if self.fallback_model not in models:
            raise ValueError("fallback_model must be one of the candidate models")
        return self


CapabilityBoundary: TypeAlias = Literal["supported", "uncertain", "unsupported", "unmatched"]


class CapabilityCandidateScore(BaseModel):
    """One classifier estimate for the current task."""

    model: str
    reason: str
    primary_rule: str = "none"
    capability_boundary: CapabilityBoundary
    p_solve: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("model", "reason", "primary_rule")
    @classmethod
    def validate_nonblank(cls, value: str, info: ValidationInfo) -> str:
        return _nonblank(value, info.field_name or "classifier field")

    @field_validator("p_solve", mode="before")
    @classmethod
    def validate_probability(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("p_solve must be a JSON number")  # noqa: TRY004  # pydantic needs ValueError
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("p_solve must be finite")
        return value


class CapabilityClassifierVerdict(BaseModel):
    """Strict structured output returned by the classifier."""

    candidates: tuple[CapabilityCandidateScore, ...] = Field(min_length=2)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_unique_models(self) -> Self:
        models: Final = tuple(candidate.model for candidate in self.candidates)
        if len(models) != len(frozenset(models)):
            raise ValueError("classifier candidate model names must be unique")
        return self


CapabilitySelectionReason: TypeAlias = Literal[
    "cheapest_qualified",
    "no_qualified_candidate",
    "missing_candidate_price",
    "classifier_error",
    "invalid_classifier_verdict",
]
