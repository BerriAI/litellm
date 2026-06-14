import os
from typing import Optional

import litellm
from litellm.types.utils import ModelInfo


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return (
        litellm.reasoning_auto_summary
        or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"
    )


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: Optional[str] = None,
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

    model_info: Optional[ModelInfo] = None
    try:
        model_info = get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )
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
