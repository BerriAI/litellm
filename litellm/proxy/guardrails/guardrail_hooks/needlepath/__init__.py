from __future__ import annotations

from collections.abc import Sequence
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
    mode: str | Sequence[str] | Mode,
) -> GuardrailEventHooks | Sequence[GuardrailEventHooks] | Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        # A real `list` is required here, not just any Sequence: CustomGuardrail's
        # dispatch (should_run_guardrail) does `isinstance(self.event_hook, list)`
        # to decide whether this hook fires for a given event. A tuple would fail
        # that check silently and the guardrail would never run for a multi-hook
        # config.
        return [GuardrailEventHooks(item) for item in mode]  # mutable-ok: see comment above
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


# guardrail_registry.py's module discovery does `isinstance(registry, dict)` before
# merging this into the central registry; a Mapping/MappingProxyType would fail that
# check silently and this guardrail would never be discovered at proxy startup.
guardrail_initializer_registry: Final = {  # mutable-ok: see comment above
    SupportedGuardrailIntegrations.NEEDLEPATH.value: initialize_guardrail,
}

# Same reason as guardrail_initializer_registry above: discovery's isinstance(dict) check.
guardrail_class_registry: Final = {  # mutable-ok: see comment above
    SupportedGuardrailIntegrations.NEEDLEPATH.value: NeedlepathGuardrail,
}
