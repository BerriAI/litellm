from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from importlib import import_module
from typing import Final, Protocol, runtime_checkable

from .config import CapabilityConfig, SwitchyardRouterConfig


class Aggregate(Protocol):
    @property
    def response(self) -> Mapping[str, object]: ...


class ModelCall(Protocol):
    @property
    def models(self) -> Sequence[str]: ...

    @property
    def request(self) -> Mapping[str, object]: ...

    async def respond(self, work: Awaitable[Aggregate]) -> None: ...


class ModelStep(Protocol):
    @property
    def call(self) -> ModelCall: ...


class Outcome(Protocol):
    @property
    def selected_model_ids(self) -> Sequence[str]: ...

    @property
    def request(self) -> Mapping[str, object]: ...

    @property
    def fallback_request(self) -> Mapping[str, object] | None: ...

    @property
    def response(self) -> Aggregate | None: ...


class DoneStep(Protocol):
    @property
    def outcome(self) -> Outcome: ...


class StepTypes(Protocol):
    CallModel: type[ModelStep]
    Done: type[DoneStep]


class ResponseTypes(Protocol):
    Agg: Callable[[Mapping[str, object]], Aggregate]


class Algorithm(Protocol):
    def run_stream(
        self,
        request: Mapping[str, object],
        models: Mapping[str, Sequence[str]],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[ModelStep | DoneStep]: ...


class ClassifierFactory(Protocol):
    def capability(self, *, config: object) -> object: ...

    def escalation(self, *, config: object) -> object: ...

    def custom(self, *, default_target: str, config: object) -> object: ...


@runtime_checkable
class NativeApi(Protocol):
    Step: StepTypes
    LlmResponse: ResponseTypes
    LlmClassifierConfig: ClassifierFactory
    ContextWindowExceededError: type[Exception]
    LibsyError: type[Exception]

    def TaskClassifierConfig(
        self,
        base_threshold: float,
        *,
        threshold_step: float,
        session_affinity: bool,
        message_hash_fallback: bool,
        recent_turn_window: int | None,
        max_output_tokens: int,
        prompt: str | None,
        response_format_type: str,
    ) -> object: ...

    def CustomClassifierConfig(
        self,
        prompt: str,
        response_schema: Mapping[str, object],
        selector: str,
        *,
        session_affinity: bool,
        message_hash_fallback: bool,
        recent_turn_window: int | None,
        max_output_tokens: int,
    ) -> object: ...

    def EscalationClassifierConfig(
        self,
        *,
        confirmations: int,
        recent_turn_window: int,
        window_message_chars: int,
        deescalation: object | None,
        max_output_tokens: int,
        prompt: str | None,
        response_format_type: str,
    ) -> object: ...

    def DeescalationConfig(
        self,
        *,
        strong_min_calls: int,
        confirmations: int,
        strong_max_calls: int | None,
        weak_cooldown_calls: int,
    ) -> object: ...

    def LlmFallback(self, *, config: object) -> object: ...

    def random(self, *, weights: Sequence[float] | None, seed: int | None) -> Algorithm: ...

    def llm_classifier(self, config: object) -> Algorithm: ...

    def stage_router(
        self,
        *,
        picker: str,
        confidence_threshold: float,
        recent_window: int | None,
        escalation_note: str | None,
        deescalation_note: str | None,
        only_on_wrong_signal_escalation: bool,
        capable_system_prompt: str | None,
        efficient_system_prompt: str | None,
        tool_semantics: Mapping[str, Sequence[str]] | None,
        classifier: object | None,
    ) -> Algorithm: ...


class NativeExtension(Protocol):
    @property
    def libsy(self) -> NativeApi: ...


@runtime_checkable
class NativeLoader(Protocol):
    def load_native(self) -> NativeExtension: ...


def load_native() -> NativeApi:
    try:
        module: Final = import_module("switchyard_rust._native")
    except ImportError as error:
        raise ImportError("Switchyard routing requires the optional dependency nemo-switchyard==0.4.0a1") from error
    if not isinstance(module, NativeLoader):
        raise TypeError("Installed Switchyard does not expose the required libsy API")
    return module.load_native().libsy


def capability_config(api: NativeApi, config: CapabilityConfig) -> object:
    return api.TaskClassifierConfig(
        base_threshold=config.base_threshold,
        threshold_step=config.threshold_step,
        session_affinity=config.session_affinity,
        message_hash_fallback=config.message_hash_fallback,
        recent_turn_window=config.recent_turn_window,
        max_output_tokens=config.max_output_tokens,
        prompt=config.prompt,
        response_format_type=config.response_format_type,
    )


def build_algorithm(api: NativeApi, config: SwitchyardRouterConfig) -> Algorithm:
    if config.algorithm == "random":
        return api.random(weights=config.weights, seed=config.seed)
    if config.algorithm == "stage":
        stage: Final = config.stage
        fallback: Final = (
            api.LlmFallback(config=capability_config(api, stage.classifier)) if stage.classifier is not None else None
        )
        return api.stage_router(
            picker=stage.picker,
            confidence_threshold=stage.confidence_threshold,
            recent_window=stage.recent_window,
            escalation_note=stage.escalation_note,
            deescalation_note=stage.deescalation_note,
            only_on_wrong_signal_escalation=stage.only_on_wrong_signal_escalation,
            capable_system_prompt=stage.capable_system_prompt,
            efficient_system_prompt=stage.efficient_system_prompt,
            tool_semantics=stage.tool_semantics,
            classifier=fallback,
        )
    if config.algorithm == "capability":
        return api.llm_classifier(api.LlmClassifierConfig.capability(config=capability_config(api, config.capability)))
    if config.algorithm == "custom":
        custom: Final = config.custom
        if custom is None:
            raise ValueError("custom routing requires custom configuration")
        return api.llm_classifier(
            api.LlmClassifierConfig.custom(
                default_target=custom.default_target,
                config=api.CustomClassifierConfig(
                    prompt=custom.prompt,
                    response_schema=custom.response_schema,
                    selector=custom.selector,
                    session_affinity=custom.session_affinity,
                    message_hash_fallback=custom.message_hash_fallback,
                    recent_turn_window=custom.recent_turn_window,
                    max_output_tokens=custom.max_output_tokens,
                ),
            )
        )
    escalation: Final = config.escalation
    release: Final = escalation.deescalation
    deescalation: Final = (
        api.DeescalationConfig(
            strong_min_calls=release.strong_min_calls,
            confirmations=release.confirmations,
            strong_max_calls=release.strong_max_calls,
            weak_cooldown_calls=release.weak_cooldown_calls,
        )
        if release is not None
        else None
    )
    return api.llm_classifier(
        api.LlmClassifierConfig.escalation(
            config=api.EscalationClassifierConfig(
                confirmations=escalation.confirmations,
                recent_turn_window=escalation.recent_turn_window,
                window_message_chars=escalation.window_message_chars,
                deescalation=deescalation,
                max_output_tokens=escalation.max_output_tokens,
                prompt=escalation.prompt,
                response_format_type=escalation.response_format_type,
            )
        )
    )
