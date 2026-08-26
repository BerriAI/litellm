import os
from typing import Final

import litellm
from litellm.types.utils import ModelInfo

OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH: Final = 64


def prompt_cache_key_from_user_id(user_id: object) -> str | None:
    if user_id is None:
        return None
    return str(user_id)[:OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH] or None


def local_model_name(model: str, custom_llm_provider: object) -> str:
    """The id the provider itself knows, for reporting back to the caller in ``message_start``."""
    return model.removeprefix(f"{custom_llm_provider}/") if isinstance(custom_llm_provider, str) else model


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return litellm.reasoning_auto_summary or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: str | None = None,
) -> str:
    """
    Normalize a reasoning effort value based on model capabilities.

    Degradation chains:
    - "max"     → max / xhigh / high
    - "xhigh"   → xhigh / high
    - "minimal" → minimal / low
    - other values pass through unchanged
    """
    if effort not in ("max", "xhigh", "minimal"):
        return effort

    from litellm.utils import get_model_info

    model_info: ModelInfo | None = None
    try:
        model_info = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        model_info = None

    if effort == "max":
        if model_info and model_info.get("supports_max_reasoning_effort"):
            return "max"
        if model_info and model_info.get("supports_xhigh_reasoning_effort"):
            return "xhigh"
        return "high"
    elif effort == "xhigh":
        if model_info and model_info.get("supports_xhigh_reasoning_effort"):
            return "xhigh"
        return "high"
    elif effort == "minimal":
        if model_info and model_info.get("supports_minimal_reasoning_effort"):
            return "minimal"
        return "low"
    return "medium"
