from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)
from litellm.types.proxy.guardrails.guardrail_hooks.decision_model import (
    DecisionModelCheck,
)

from .decision_model import DecisionModelGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

_CHECKS_ADAPTER: Final = TypeAdapter(list[DecisionModelCheck])


def _coerce_event_hook(
    mode: str | Sequence[str] | Mode,
) -> GuardrailEventHooks | tuple[GuardrailEventHooks, ...] | Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, str):
        return GuardrailEventHooks(mode)
    return tuple(GuardrailEventHooks(item) for item in mode)


def _parse_checks(raw_checks: object) -> tuple[DecisionModelCheck, ...]:
    try:
        checks: Final = _CHECKS_ADAPTER.validate_python(raw_checks)
    except ValidationError as error:
        raise ValueError(f"decision_model guardrail has invalid checks: {error}") from error
    return tuple(checks)


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> DecisionModelGuardrail:
    import litellm

    decision_model: Final = litellm_params.decision_model
    if not decision_model:
        raise ValueError("decision_model guardrail requires decision_model in litellm_params")

    checks: Final = _parse_checks(litellm_params.checks)

    _callback: Final = DecisionModelGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        decision_model=decision_model,
        checks=checks,
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
        unreachable_fallback=(
            litellm_params.unreachable_fallback
            if "unreachable_fallback" in litellm_params.model_fields_set
            else "fail_closed"
        ),
        max_input_chars=litellm_params.max_input_chars or 24000,
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]  # callback manager is untyped
        _callback
    )
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.DECISION_MODEL.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.DECISION_MODEL.value: DecisionModelGuardrail,
}


__all__ = [
    "DecisionModelGuardrail",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
]
