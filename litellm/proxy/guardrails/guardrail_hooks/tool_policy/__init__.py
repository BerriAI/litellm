import litellm
from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.tool_policy.tool_policy_guardrail import (
        ToolPolicyGuardrail,
    )

    _callback = ToolPolicyGuardrail(
        guardrail_name=guardrail.get("guardrail_name", "tool_policy"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback
