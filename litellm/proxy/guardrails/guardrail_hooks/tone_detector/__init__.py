from typing import TYPE_CHECKING, Optional

import litellm
from litellm.proxy.guardrails.guardrail_hooks.tone_detector.tone_detector import (
    ToneDetectorGuardrail,
)
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    llm_router: Optional["Router"] = None,
):
    """Initialize the Tone Detector Guardrail."""
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Tone Detector: guardrail_name is required")

    tone_guardrail = ToneDetectorGuardrail(
        guardrail_name=guardrail_name,
        blocked_phrases=getattr(litellm_params, "blocked_phrases", None),
        safe_phrases=getattr(litellm_params, "safe_phrases", None),
        event_hook=litellm_params.mode,  # type: ignore
        default_on=litellm_params.default_on or False,
        violation_message_template=getattr(
            litellm_params, "violation_message_template", None
        ),
    )

    litellm.logging_callback_manager.add_litellm_callback(tone_guardrail)
    return tone_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.TONE_DETECTOR.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.TONE_DETECTOR.value: ToneDetectorGuardrail,
}
