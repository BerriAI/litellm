from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .akto import AktoGuardrail


if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _akto_callback = AktoGuardrail(
        akto_base_url=getattr(litellm_params, "akto_base_url", None),
        akto_api_key=getattr(litellm_params, "akto_api_key", None),
        unreachable_fallback=getattr(litellm_params, "unreachable_fallback", "fail_closed"),
        guardrail_timeout=getattr(litellm_params, "guardrail_timeout", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_akto_callback)
    return _akto_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AKTO.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.AKTO.value: AktoGuardrail,
}
