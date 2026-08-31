from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .llama_guard import LlamaGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    model: Final = litellm_params.get("model")
    if not model:
        raise ValueError("llama_guard guardrail requires `model` in litellm_params")

    _guardrail: Final = LlamaGuardGuardrail(
        model=model,
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_base=litellm_params.get("api_base"),
        api_key=litellm_params.get("api_key"),
        categories=litellm_params.get("categories"),
        unsafe_content_categories=litellm_params.get("unsafe_content_categories"),
        event_hook=litellm_params.get("mode"),
        default_on=litellm_params.get("default_on", False),
    )
    litellm.logging_callback_manager.add_litellm_callback(_guardrail)

    return _guardrail


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.LLAMA_GUARD.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.LLAMA_GUARD.value: LlamaGuardGuardrail,
}


__all__ = [
    "LlamaGuardGuardrail",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
]
