from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)

from .needlepath import NeedlepathGuardrail

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
    """Read a knob from ``optional_params`` first, then from the top level.

    Both spellings are accepted because both appear in the wild: the nested
    ``optional_params`` block is the documented form, and several deployments
    set guardrail knobs flat alongside ``api_key``.
    """
    if optional_params is not None:
        value: Final = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> NeedlepathGuardrail:
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    _callback: Final = NeedlepathGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        select_tool_outputs=_get_optional_value(litellm_params, optional_params, "select_tool_outputs"),
        select_history=_get_optional_value(litellm_params, optional_params, "select_history"),
        select_system=_get_optional_value(litellm_params, optional_params, "select_system"),
        min_chars_to_select=_get_optional_value(litellm_params, optional_params, "min_chars_to_select"),
        max_context_tokens=_get_optional_value(litellm_params, optional_params, "max_context_tokens"),
        operating_point=_get_optional_value(litellm_params, optional_params, "operating_point"),
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]  # callback manager is untyped
        _callback
    )
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.NEEDLEPATH.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.NEEDLEPATH.value: NeedlepathGuardrail,
}
