from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .javelin import JavelinGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if litellm_params.guard_name is None:
        raise Exception(
            "JavelinGuardrailException - Please pass the Javelin guard name via 'litellm_params::guard_name'"
        )

    _javelin_callback = JavelinGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        javelin_guard_name=litellm_params.guard_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
        api_version=litellm_params.api_version or "v1",
        config=litellm_params.config,
        metadata=litellm_params.metadata,
        application=litellm_params.application,
    )
    litellm.logging_callback_manager.add_litellm_callback(_javelin_callback)

    return _javelin_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.JAVELIN.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.JAVELIN.value: JavelinGuardrail,
}
