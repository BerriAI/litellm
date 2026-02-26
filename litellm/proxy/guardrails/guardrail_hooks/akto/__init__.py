from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .akto import AktoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _akto_callback = AktoGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        sync_mode=getattr(litellm_params, "sync_mode", None),
        akto_account_id=getattr(litellm_params, "akto_account_id", None),
        akto_vxlan_id=getattr(litellm_params, "akto_vxlan_id", None),
        unreachable_fallback=getattr(
            litellm_params, "unreachable_fallback", "fail_closed"
        ),
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
