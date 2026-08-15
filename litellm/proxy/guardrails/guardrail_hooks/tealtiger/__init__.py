from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .tealtiger import TealTigerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _tealtiger_callback: Final = TealTigerGuardrail(
        policies=getattr(litellm_params, "policies", None),
        policy_mode=getattr(litellm_params, "policy_mode", "ENFORCE"),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_tealtiger_callback)
    return _tealtiger_callback


guardrail_initializer_registry: Final = MappingProxyType(
    {
        SupportedGuardrailIntegrations.TEALTIGER.value: initialize_guardrail,
    }
)

guardrail_class_registry: Final = MappingProxyType(
    {
        SupportedGuardrailIntegrations.TEALTIGER.value: TealTigerGuardrail,
    }
)
