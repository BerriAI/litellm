from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)
from litellm.types.proxy.guardrails.guardrail_hooks.typesafe import (
    TypeSafeGuardrailOptionalParams,
)

from .typesafe import TypeSafeGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _coerce_event_hook(
    mode: str | list[str] | Mode,
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [  # mutable-ok: CustomGuardrail event_hook contract wants a list
            GuardrailEventHooks(item) for item in mode
        ]
    return GuardrailEventHooks(mode)


def _optional_params(litellm_params: LitellmParams) -> TypeSafeGuardrailOptionalParams:
    value: Final = litellm_params.optional_params
    if isinstance(value, TypeSafeGuardrailOptionalParams):
        return value
    if isinstance(value, BaseModel):
        return TypeSafeGuardrailOptionalParams.model_validate(value.model_dump())
    return TypeSafeGuardrailOptionalParams()


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> TypeSafeGuardrail:
    import litellm

    optional_params: Final = _optional_params(litellm_params)

    _callback: Final = TypeSafeGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        model=litellm_params.model,
        relevance_threshold=optional_params.relevance_threshold,
        min_chars_to_evaluate=optional_params.min_chars_to_evaluate,
        max_result_chars_in_state=optional_params.max_result_chars_in_state,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
        unreachable_fallback=(
            litellm_params.unreachable_fallback if "unreachable_fallback" in litellm_params.model_fields_set else None
        ),
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]  # callback manager is untyped
        _callback
    )
    return _callback


guardrail_initializer_registry: Final = {  # mutable-ok: guardrail_registry discovery checks isinstance(registry, dict)
    SupportedGuardrailIntegrations.TYPESAFE.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: guardrail_registry discovery checks isinstance(registry, dict)
    SupportedGuardrailIntegrations.TYPESAFE.value: TypeSafeGuardrail,
}
