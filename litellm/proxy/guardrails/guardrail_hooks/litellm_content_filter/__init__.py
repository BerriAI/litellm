from typing import TYPE_CHECKING, Optional

import litellm
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
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
    """
    Initialize the Content Filter Guardrail.

    Args:
        litellm_params: Guardrail configuration parameters
        guardrail: Guardrail metadata

    Returns:
        Initialized ContentFilterGuardrail instance
    """
    guardrail_name = guardrail.get("guardrail_name")

    if not guardrail_name:
        raise ValueError("Content Filter: guardrail_name is required")

    content_filter_guardrail = ContentFilterGuardrail(
        guardrail_name=guardrail_name,
        guardrail_id=guardrail.get("guardrail_id"),
        policy_template=guardrail.get("policy_template"),
        patterns=litellm_params.patterns,
        blocked_words=litellm_params.blocked_words,
        blocked_words_file=litellm_params.blocked_words_file,
        event_hook=litellm_params.mode,  # type: ignore
        default_on=litellm_params.default_on or False,
        categories=getattr(litellm_params, "categories", None),
        severity_threshold=getattr(litellm_params, "severity_threshold", "medium"),
        llm_router=llm_router,
        image_model=getattr(litellm_params, "image_model", None),
    )

    litellm.logging_callback_manager.add_litellm_callback(content_filter_guardrail)

    return content_filter_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.LITELLM_CONTENT_FILTER.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.LITELLM_CONTENT_FILTER.value: ContentFilterGuardrail,
}
