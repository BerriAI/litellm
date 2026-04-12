"""
Semantic Guard guardrail — embedding-based prompt injection detection.

Uses semantic-router to match user prompts against known attack patterns.
"""

from typing import TYPE_CHECKING, Optional

import litellm
from litellm.constants import (
    DEFAULT_SEMANTIC_GUARD_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_GUARD_SIMILARITY_THRESHOLD,
)
from litellm.proxy.guardrails.guardrail_hooks.semantic_guard.semantic_guard import (
    SemanticGuardrail,
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
    Initialize the Semantic Guard guardrail.

    Args:
        litellm_params: Guardrail configuration parameters
        guardrail: Guardrail metadata
        llm_router: LiteLLM Router instance (required for embeddings)

    Returns:
        Initialized SemanticGuardrail instance
    """
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("SemanticGuard: guardrail_name is required")

    if llm_router is None:
        raise ValueError(
            "SemanticGuard requires llm_router for embeddings. "
            "Configure a model_list with an embedding model."
        )

    semantic_guardrail = SemanticGuardrail(
        guardrail_name=guardrail_name,
        llm_router=llm_router,
        embedding_model=getattr(litellm_params, "embedding_model", None)
        or DEFAULT_SEMANTIC_GUARD_EMBEDDING_MODEL,
        similarity_threshold=getattr(litellm_params, "similarity_threshold", None)
        or DEFAULT_SEMANTIC_GUARD_SIMILARITY_THRESHOLD,
        route_templates=getattr(litellm_params, "route_templates", None),
        custom_routes_file=getattr(litellm_params, "custom_routes_file", None),
        custom_routes=getattr(litellm_params, "custom_routes", None),
        on_flagged_action=getattr(litellm_params, "on_flagged_action", "block"),
        event_hook=litellm_params.mode,  # type: ignore
        default_on=litellm_params.default_on or False,
    )

    litellm.logging_callback_manager.add_litellm_callback(semantic_guardrail)

    return semantic_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.SEMANTIC_GUARD.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.SEMANTIC_GUARD.value: SemanticGuardrail,
}
