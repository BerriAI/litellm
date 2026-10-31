from __future__ import annotations

from collections.abc import Mapping
from itertools import chain
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CapabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_threshold: float = Field(default=0.5, ge=0, le=1)
    threshold_step: float = Field(default=0, ge=0, le=1)
    session_affinity: bool = False
    message_hash_fallback: bool = False
    recent_turn_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    prompt: str | None = None
    response_format_type: Literal["json_schema", "json_object"] = "json_schema"

    @model_validator(mode="after")
    def validate_affinity(self) -> CapabilityConfig:
        if self.message_hash_fallback and not self.session_affinity:
            raise ValueError("message_hash_fallback requires session_affinity")
        return self


class DeescalationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strong_min_calls: int = Field(gt=0)
    confirmations: int = Field(gt=0)
    strong_max_calls: int | None = Field(default=None, gt=0)
    weak_cooldown_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_call_bounds(self) -> DeescalationConfig:
        if self.strong_max_calls is not None and self.strong_max_calls < self.strong_min_calls:
            raise ValueError("strong_max_calls must be at least strong_min_calls")
        return self


class EscalationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    confirmations: int = Field(default=2, gt=0)
    recent_turn_window: int = Field(default=28, gt=0)
    window_message_chars: int = Field(default=500, ge=50)
    max_output_tokens: int = Field(default=4096, gt=0)
    prompt: str | None = None
    response_format_type: Literal["json_schema", "json_object"] = "json_schema"
    deescalation: DeescalationConfig | None = None


class CustomConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = Field(min_length=1)
    response_schema: Mapping[str, object]
    selector: str = Field(min_length=1)
    default_target: str = Field(min_length=1)
    session_affinity: bool = False
    message_hash_fallback: bool = False
    recent_turn_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def validate_affinity(self) -> CustomConfig:
        if self.message_hash_fallback and not self.session_affinity:
            raise ValueError("message_hash_fallback requires session_affinity")
        return self


class StageConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    picker: Literal["efficient_first", "capable_first"] = "efficient_first"
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    recent_window: int | None = Field(default=None, gt=0)
    escalation_note: str | None = None
    deescalation_note: str | None = None
    only_on_wrong_signal_escalation: bool = True
    capable_system_prompt: str | None = None
    efficient_system_prompt: str | None = None
    tool_semantics: Mapping[str, tuple[str, ...]] | None = None
    classifier: CapabilityConfig | None = None

    @model_validator(mode="after")
    def validate_notes_and_tools(self) -> StageConfig:
        if self.deescalation_note is not None and self.escalation_note is None:
            raise ValueError("deescalation_note requires escalation_note")
        if self.tool_semantics is not None and set(self.tool_semantics) - {"observe", "mutate", "plan", "new"}:
            raise ValueError("tool_semantics supports observe, mutate, plan, and new categories")
        return self


class SwitchyardRouterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["random", "capability", "custom", "escalation", "stage"] = "capability"
    models: Mapping[str, tuple[str, ...]]
    weights: tuple[float, ...] | None = None
    seed: int | None = Field(default=None, ge=0, le=18446744073709551615)
    fallback_enabled: bool = True
    capability: CapabilityConfig = Field(default_factory=CapabilityConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    custom: CustomConfig | None = None
    stage: StageConfig = Field(default_factory=StageConfig)

    @model_validator(mode="after")
    def validate_models(self) -> SwitchyardRouterConfig:
        if any(
            not group or not names or any(not name.strip() for name in names) for group, names in self.models.items()
        ):
            raise ValueError("model groups and model names must be nonempty")
        required: Final = self.required_groups()
        if any(group not in self.models for group in required):
            raise ValueError(f"{self.algorithm} requires model groups {', '.join(required)}")
        if self.algorithm == "custom" and self.custom is None:
            raise ValueError("custom routing requires custom configuration")
        if self.custom is not None and self.custom.default_target not in self.models:
            raise ValueError("custom.default_target must name a configured model group")
        if self.weights is not None:
            import math

            if self.algorithm != "random":
                raise ValueError("weights are only supported for random routing")
            if len(self.weights) != len(self.models.get("any", ())):
                raise ValueError("weights must match the any model group")
            if any(not math.isfinite(weight) or weight < 0 for weight in self.weights) or not any(self.weights):
                raise ValueError("weights must be finite, nonnegative, and have a positive total")
        return self

    def required_groups(self) -> tuple[str, ...]:
        if self.algorithm == "random":
            return ("any",)
        if self.algorithm == "custom":
            return ("judge",)
        if self.algorithm == "stage" and self.stage.classifier is None:
            return ("efficient", "capable")
        return ("efficient", "capable", "judge")

    def runtime_models(self) -> Mapping[str, tuple[str, ...]]:
        if self.algorithm == "random":
            return self.models
        target_groups: Final = tuple(names for group, names in self.models.items() if group not in ("judge", "any"))
        targets: Final = tuple(dict.fromkeys(chain(self.models.get("any", ()), chain.from_iterable(target_groups))))
        return {**self.models, "any": targets}
