from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
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
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def _get_optional_value(litellm_params: LitellmParams, optional_params: object | None, attribute_name: str) -> object:
    if optional_params is not None:
        value: Final = getattr(optional_params, attribute_name, None)
        if value is not None:
            return cast(object, value)
    return cast(object, getattr(litellm_params, attribute_name, None))


def _optional_float(litellm_params: LitellmParams, optional_params: object | None, attribute_name: str) -> float | None:
    value: Final = _get_optional_value(litellm_params, optional_params, attribute_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(litellm_params: LitellmParams, optional_params: object | None, attribute_name: str) -> int | None:
    value: Final = _get_optional_value(litellm_params, optional_params, attribute_name)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> TypeSafeGuardrail:
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    _callback: Final = TypeSafeGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        model=litellm_params.model,
        relevance_threshold=_optional_float(litellm_params, optional_params, "relevance_threshold"),
        min_chars_to_evaluate=_optional_int(litellm_params, optional_params, "min_chars_to_evaluate"),
        max_result_chars_in_state=_optional_int(litellm_params, optional_params, "max_result_chars_in_state"),
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


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.TYPESAFE.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.TYPESAFE.value: TypeSafeGuardrail,
}
