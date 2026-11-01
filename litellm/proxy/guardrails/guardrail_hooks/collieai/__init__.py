from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.collieai import DEFAULT_API_BASE

from .collieai import CollieAiGuardrail

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

# The package's own settings a config entry may carry beside LiteLLM's. The package validates their values and
# names a bad one at startup
PACKAGE_SETTINGS: Final = frozenset(
    {
        "enforcement_mode",
        "deadline_ms",
        "network_margin_ms",
        "max_body_bytes",
        "forward_conversation_id",
        "summary_interval_s",
        "synthetic_interval_s",
    }
)
_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})


def package_settings(extras: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in extras.items() if key in PACKAGE_SETTINGS})


def initialize_guardrail(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
    guardrail_cls: Callable[..., CustomGuardrail] = CollieAiGuardrail,
) -> CustomGuardrail:
    import litellm

    extras: Final = litellm_params.model_extra or _NO_EXTRAS
    _callback: Final = guardrail_cls(
        api_base=litellm_params.api_base or DEFAULT_API_BASE,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        **package_settings(extras),
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.COLLIEAI.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.COLLIEAI.value: CollieAiGuardrail,
}
