from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .generic_guardrail_api import GenericGuardrailAPI

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

# Runtime initialization is owned by the Rust dispatcher in
# guardrail_hooks/guardrail_v2/__init__.py, which routes to Rust and calls
# initialize_python_guardrail below as a fallback. This module only exposes the
# class registry (UI config-field schemas, and GenericGuardrailAPI remains the
# base class for guardrails like qohash) and that fallback initializer.


def initialize_python_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
):
    import litellm

    callback = GenericGuardrailAPI(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        headers=getattr(litellm_params, "headers", None),
        additional_provider_specific_params=getattr(
            litellm_params, "additional_provider_specific_params", {}
        ),
        unreachable_fallback=getattr(
            litellm_params, "unreachable_fallback", "fail_closed"
        ),
        extra_headers=getattr(litellm_params, "extra_headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


guardrail_class_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: GenericGuardrailAPI,
}
