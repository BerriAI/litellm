from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import GuardrailEventHooks, Mode

from .patronus_ark import PatronusArkGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> PatronusArkGuardrail:
    import litellm

    mode: Final = litellm_params.mode
    event_hook: Final = (
        mode
        if isinstance(mode, Mode)
        else ([GuardrailEventHooks(item) for item in mode] if isinstance(mode, list) else GuardrailEventHooks(mode))
    )
    callback: Final = PatronusArkGuardrail(
        ark=litellm_params.ark,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=event_hook,
        default_on=litellm_params.default_on or False,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        timeout=litellm_params.timeout,
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


guardrail_initializer_registry: Final = {"patronus_ark": initialize_guardrail}
guardrail_class_registry: Final = {"patronus_ark": PatronusArkGuardrail}
