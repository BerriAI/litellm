from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, model_validator

if TYPE_CHECKING:
    from switchyard_rust.libsy import Algorithm, TaskClassifierConfig


class LlmClassifierConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["llm"] = "llm"
    base_threshold: float = Field(default=0.5, ge=0, le=1)
    threshold_step: float = Field(default=0.0, ge=0, le=1)
    recent_turn_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    response_format_type: Literal["json_schema", "json_object"] = "json_schema"
    prompt: str | None = None


class DecisionClassifierConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["decision"]
    cutoff: float = Field(default=0.6, ge=0, le=1)
    instructions: str | Mapping[str, object] | None = None
    evidence: Mapping[str, object] | None = None


_KIND: Final = TypeAdapter(Mapping[str, object])


def _classifier_kind(value: object) -> str:
    match value:
        case LlmClassifierConfig() | DecisionClassifierConfig():
            return value.kind
        case Mapping():
            return str(_KIND.validate_python(value).get("kind", "llm"))
        case _:
            return "llm"


ClassifierConfig: TypeAlias = Annotated[
    Annotated[LlmClassifierConfig, Tag("llm")] | Annotated[DecisionClassifierConfig, Tag("decision")],
    Discriminator(_classifier_kind),
]


class StageOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    picker: Literal["efficient_first", "capable_first"] = "efficient_first"
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    recent_window: int | None = Field(default=None, gt=0)
    tool_semantics: Mapping[str, tuple[str, ...]] | None = None
    only_on_wrong_signal_escalation: bool = True


class StageModels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    capable: str = Field(min_length=1)
    efficient: str = Field(min_length=1)
    judge: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def reject_recursive_targets(self) -> StageModels:
        if any(name.startswith("auto_router/") for name in (self.capable, self.efficient, self.judge or "")):
            raise ValueError("Switchyard targets must be concrete model groups, not auto-router markers")
        if self.capable == self.efficient:
            raise ValueError("The capable and efficient models must be different model groups")
        return self


class SwitchyardRouterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["stage_router"] = "stage_router"
    models: StageModels
    options: StageOptions = Field(default_factory=StageOptions)
    classifier: ClassifierConfig | None = None

    @model_validator(mode="after")
    def validate_judge(self) -> SwitchyardRouterConfig:
        if (self.classifier is None) != (self.models.judge is None):
            raise ValueError("A judge model is required exactly when the classifier is enabled")
        return self

    @property
    def confidence_threshold(self) -> float:
        if self.options.confidence_threshold is not None:
            return self.options.confidence_threshold
        return 0.7 if self.classifier is not None else 0.5

    def runtime_models(self) -> Mapping[str, tuple[str, ...]]:
        return {
            "any": (self.models.capable, self.models.efficient),
            "capable": (self.models.capable,),
            "efficient": (self.models.efficient,),
            **({"judge": (self.models.judge,)} if self.models.judge is not None else {}),
        }


def _task_classifier(config: SwitchyardRouterConfig) -> TaskClassifierConfig | None:
    from switchyard_rust.libsy import TaskClassifierConfig

    match config.classifier:
        case None:
            return None
        case LlmClassifierConfig() as llm:
            return TaskClassifierConfig(
                base_threshold=llm.base_threshold,
                threshold_step=llm.threshold_step,
                recent_turn_window=llm.recent_turn_window,
                max_output_tokens=llm.max_output_tokens,
                response_format_type=llm.response_format_type,
                prompt=llm.prompt,
            )
        case DecisionClassifierConfig() as decision:
            return TaskClassifierConfig.decision(
                cutoff=decision.cutoff,
                candidates={"capable": config.models.capable, "efficient": config.models.efficient},
                evidence=dict(decision.evidence or {}),
                instructions=decision.instructions,
            )


def build_algorithm(config: SwitchyardRouterConfig) -> Algorithm:
    from switchyard_rust.libsy import LlmFallback, stage_router

    classifier: Final = _task_classifier(config)
    return stage_router(
        picker=config.options.picker,
        confidence_threshold=config.confidence_threshold,
        recent_window=config.options.recent_window,
        tool_semantics=dict(config.options.tool_semantics) if config.options.tool_semantics is not None else None,
        only_on_wrong_signal_escalation=config.options.only_on_wrong_signal_escalation,
        classifier=LlmFallback(config=classifier) if classifier is not None else None,
    )


def validate_switchyard_config_write(config: Mapping[str, object] | None) -> str | None:
    if config is None:
        return None
    try:
        build_algorithm(SwitchyardRouterConfig.model_validate(config))
    except ImportError:
        return "Switchyard requires the optional dependency. Install litellm[switchyard] (nemo-switchyard==0.4.0a1)."
    except (ValueError, TypeError) as error:
        return f"Invalid switchyard_router_config: {error}"
    return None
