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
    return (
        litellm.reasoning_auto_summary
        or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"
    )


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: str | None = None,
) -> str:
    """
    Normalize a reasoning effort value based on model capabilities.

    Degradation only happens when a capability flag is **explicitly False**.
    Unknown models (flag absent/None) pass the effort through unchanged so
    third-party Anthropic-compatible deployments are never silently downgraded.

    Degradation chains (only when flag is False):
    - "max"     → xhigh (if xhigh not False) / high
    - "xhigh"   → high
    - "minimal" → low
    - other values pass through unchanged
    """
    if effort not in ("max", "xhigh", "minimal"):
        return effort

    from litellm.utils import get_model_info

    model_info: ModelInfo | None = None
    try:
        model_info = get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )
    except Exception:
        model_info = None

    def _flag(key: str) -> Optional[bool]:
        """Return the flag value, or None when not declared."""
        if not model_info:
            return None
        return model_info.get(key)

    if effort == "max":
        if _flag("supports_max_reasoning_effort") is False:
            # max explicitly unsupported — try xhigh, then high
            if _flag("supports_xhigh_reasoning_effort") is False:
                return "high"
            return "xhigh"
        return "max"
    elif effort == "xhigh":
        if _flag("supports_xhigh_reasoning_effort") is False:
            return "high"
        return "xhigh"
    elif effort == "minimal":
        if _flag("supports_minimal_reasoning_effort") is False:
            return "low"
        return "minimal"
    return "medium"
