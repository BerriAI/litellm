from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .reco import RecoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    _reco_callback: Final = RecoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        reco_tenant_id=getattr(optional_params, "reco_tenant_id", None),
        api_base=getattr(optional_params, "api_base", None),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_reco_callback)

    return _reco_callback


guardrail_initializer_registry: Final = {  # mutable-ok: guardrail auto-discovery requires an actual dict instance
    SupportedGuardrailIntegrations.RECO.value: initialize_guardrail,
}


guardrail_class_registry: Final = {  # mutable-ok: guardrail auto-discovery requires an actual dict instance
    SupportedGuardrailIntegrations.RECO.value: RecoGuardrail,
}
