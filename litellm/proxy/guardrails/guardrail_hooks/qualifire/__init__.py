from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .qualifire import QualifireGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _qualifire_callback = QualifireGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        evaluation_id=getattr(litellm_params, "evaluation_id", None),
        prompt_injections=getattr(litellm_params, "prompt_injections", None),
        hallucinations_check=getattr(litellm_params, "hallucinations_check", None),
        grounding_check=getattr(litellm_params, "grounding_check", None),
        pii_check=getattr(litellm_params, "pii_check", None),
        content_moderation_check=getattr(litellm_params, "content_moderation_check", None),
        tool_selection_quality_check=getattr(litellm_params, "tool_selection_quality_check", None),
        assertions=getattr(litellm_params, "assertions", None),
        on_flagged=getattr(litellm_params, "on_flagged", "block"),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_qualifire_callback)

    return _qualifire_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.QUALIFIRE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.QUALIFIRE.value: QualifireGuardrail,
}
