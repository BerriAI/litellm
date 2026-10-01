from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

from litellm.types.guardrails import GuardrailEventHooks, Mode, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import RealmLabsGuardrailOptionalParams

from .realmlabs import RealmLabsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

__all__ = ("RealmLabsGuardrail",)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> RealmLabsGuardrail:
    """Build the guardrail from its ``config.yaml`` entry and register it; found via the registries below."""
    import litellm

    settings: Final = _resolved_params(litellm_params)
    _realmlabs_callback: Final = RealmLabsGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        probes=settings.probes,
        hazard_threshold=settings.hazard_threshold,
        pii=settings.pii,
        pii_mask=settings.pii_mask,
        block_on_error=settings.block_on_error,
        enable_thinking=settings.enable_thinking,
        timeout=settings.timeout,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]  # callback manager is untyped
        _realmlabs_callback
    )
    return _realmlabs_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.REALMLABS.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.REALMLABS.value: RealmLabsGuardrail,
}


def _coerce_event_hook(
    mode: str | list[str] | Mode,  # mutable-ok: mirrors LitellmParams.mode
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:  # mutable-ok: mirrors CustomGuardrail's event_hook
    """Convert the ``mode`` strings from ``config.yaml`` into the enum values ``CustomGuardrail`` expects."""
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def _resolved_params(litellm_params: "LitellmParams") -> RealmLabsGuardrailOptionalParams:
    """Resolve explicit nested values, then top-level values, then RealmLabs defaults.

    Excluding unset fields prevents another guardrail's parsed defaults from overriding RealmLabs settings.
    Null nested values fall back to the top level; false, zero, and empty lists remain explicit overrides.
    """
    top_level: Final = RealmLabsGuardrailOptionalParams.model_validate(litellm_params.model_dump(exclude_unset=True))
    nested: Final = litellm_params.optional_params
    if not isinstance(nested, BaseModel):
        return top_level
    return RealmLabsGuardrailOptionalParams.model_validate(
        MappingProxyType({**top_level.model_dump(), **nested.model_dump(exclude_unset=True, exclude_none=True)})
    )
