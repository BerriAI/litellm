from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .hiddenlayer import HiddenlayerGuardrail, HiddenlayerGuardrailV2

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    version: Final = litellm_params.version
    guardrail_cls: Final = HiddenlayerGuardrail if not version or version < 2 else HiddenlayerGuardrailV2
    _hiddenlayer_callback: Final = guardrail_cls(
        api_base=litellm_params.api_base,
        api_id=litellm_params.api_id,
        api_key=litellm_params.api_key,
        auth_url=litellm_params.auth_url,
        ssl_verify=litellm_params.ssl_verify,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        timeout=litellm_params.timeout,
    )

    litellm.logging_callback_manager.add_litellm_callback(_hiddenlayer_callback)
    return _hiddenlayer_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.HIDDENLAYER.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.HIDDENLAYER.value: HiddenlayerGuardrail,
}
