"""Block Code Execution guardrail: blocks or masks fenced code blocks by language."""

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations

from .block_code_execution import BlockCodeExecutionGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

# Default: run on both request and response (and during_call is supported too)
DEFAULT_EVENT_HOOKS = [
    GuardrailEventHooks.pre_call.value,
    GuardrailEventHooks.post_call.value,
]


def _get_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: Any = None,
) -> Any:
    """Get a param from litellm_params, with fallback to raw guardrail litellm_params (for extra fields not on LitellmParams)."""
    value = getattr(litellm_params, key, default)
    if value is not None:
        return value
    raw = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> BlockCodeExecutionGuardrail:
    """Initialize the Block Code Execution guardrail from config."""
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError(
            "Block Code Execution guardrail requires a guardrail_name"
        )

    blocked_languages: Optional[List[str]] = cast(
        Optional[List[str]],
        _get_param(litellm_params, guardrail, "blocked_languages"),
    )
    action = cast(
        Literal["block", "mask"],
        _get_param(litellm_params, guardrail, "action", "block"),
    )
    confidence_threshold = float(
        cast(
            Union[int, float, str],
            _get_param(litellm_params, guardrail, "confidence_threshold", 0.5),
        )
    )
    detect_execution_intent = bool(
        _get_param(litellm_params, guardrail, "detect_execution_intent", True)
    )
    mode = _get_param(litellm_params, guardrail, "mode")
    event_hook = cast(
        Optional[Union[Literal["pre_call", "post_call", "during_call"], List[str]]],
        mode if mode is not None else DEFAULT_EVENT_HOOKS,
    )

    instance = BlockCodeExecutionGuardrail(
        guardrail_name=guardrail_name,
        blocked_languages=blocked_languages,
        action=action,
        confidence_threshold=confidence_threshold,
        detect_execution_intent=detect_execution_intent,
        event_hook=event_hook,
        default_on=bool(_get_param(litellm_params, guardrail, "default_on", False)),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.BLOCK_CODE_EXECUTION.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.BLOCK_CODE_EXECUTION.value: BlockCodeExecutionGuardrail,
}

__all__ = [
    "BlockCodeExecutionGuardrail",
    "initialize_guardrail",
]
