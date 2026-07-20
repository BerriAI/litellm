from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .agentguards import AgentGuardsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _agentguards_callback = AgentGuardsGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        # use_case / tenant_id / fail_closed are extra litellm_params
        # (LitellmParams has extra="allow"), read via getattr.
        use_case=getattr(litellm_params, "use_case", None),
        tenant_id=getattr(litellm_params, "tenant_id", None),
        fail_closed=getattr(litellm_params, "fail_closed", None),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_agentguards_callback)

    return _agentguards_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AGENTGUARDS.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.AGENTGUARDS.value: AgentGuardsGuardrail,
}
