from typing import TYPE_CHECKING, Literal, cast

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .generic_guardrail_api import GenericGuardrailAPI

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if litellm_params.api_version is not None and litellm_params.api_version not in [
        "beta",
        "v1",
    ]:
        raise ValueError("Invalid API version. Please use 'beta' or 'v1'")

    _generic_guardrail_api_callback = GenericGuardrailAPI(
        api_base=litellm_params.api_base,
        headers=getattr(litellm_params, "headers", None),
        additional_provider_specific_params=getattr(
            litellm_params, "additional_provider_specific_params", {}
        ),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        api_version=cast(Literal["beta", "v1"], litellm_params.api_version or "beta"),
    )

    litellm.logging_callback_manager.add_litellm_callback(
        _generic_guardrail_api_callback
    )
    return _generic_guardrail_api_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: GenericGuardrailAPI,
}
