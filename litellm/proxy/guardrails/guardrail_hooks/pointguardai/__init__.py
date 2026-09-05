from typing import TYPE_CHECKING, Final, Protocol

from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)

from .pointguardai import PointGuardAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


class _CallbackRegistrar(Protocol):
    def add_litellm_callback(self, callback: PointGuardAIGuardrail) -> None: ...


def _resolve_secret_reference(value: str | None) -> str | None:
    if value is not None and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def _coerce_event_hook(
    mode: str | list[str] | Mode,  # mutable-ok: mirrors the LiteLLM mode configuration contract
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:  # mutable-ok: inherited hook API requires a list
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    callback_manager: _CallbackRegistrar | None = None,
) -> PointGuardAIGuardrail:
    import litellm

    configured_fields_value: Final = getattr(litellm_params, "model_fields_set", None)
    configured_fields: Final = (
        configured_fields_value
        if configured_fields_value is not None
        else getattr(litellm_params, "__fields_set__", frozenset())
    )
    unreachable_fallback: Final = (
        litellm_params.unreachable_fallback if "unreachable_fallback" in configured_fields else "fail_closed"
    )

    pointguardai_guardrail: Final = PointGuardAIGuardrail(
        guardrail_name=guardrail.get("guardrail_name"),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        org_code=_resolve_secret_reference(litellm_params.org_code),
        policy_config_name=_resolve_secret_reference(litellm_params.policy_config_name),
        unreachable_fallback=unreachable_fallback,
        default_on=litellm_params.default_on or False,
        event_hook=_coerce_event_hook(litellm_params.mode),
    )

    if callback_manager is None:
        litellm.logging_callback_manager.add_litellm_callback(pointguardai_guardrail)
    else:
        callback_manager.add_litellm_callback(pointguardai_guardrail)
    return pointguardai_guardrail


guardrail_initializer_registry: Final = {  # mutable-ok: LiteLLM registry contract requires a dictionary
    SupportedGuardrailIntegrations.POINTGUARDAI.value: initialize_guardrail,
}


guardrail_class_registry: Final = {  # mutable-ok: LiteLLM registry contract requires a dictionary
    SupportedGuardrailIntegrations.POINTGUARDAI.value: PointGuardAIGuardrail,
}


__all__: Final = (
    "PointGuardAIGuardrail",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
)
