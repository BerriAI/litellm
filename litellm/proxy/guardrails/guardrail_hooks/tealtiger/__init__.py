from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .config import parse_config
from .tealtiger import TealTigerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> TealTigerGuardrail:
    import litellm

    config: Final = parse_config(
        getattr(litellm_params, "policies", None), getattr(litellm_params, "policy_mode", None)
    )
    _tealtiger_callback: Final = TealTigerGuardrail(
        policies=config.policies,
        policy_mode=config.policy_mode,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_tealtiger_callback)
    return _tealtiger_callback


# Plain dicts on purpose: guardrail_registry.py discovers hooks with isinstance(registry, dict)
guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.TEALTIGER.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.TEALTIGER.value: TealTigerGuardrail,
}
