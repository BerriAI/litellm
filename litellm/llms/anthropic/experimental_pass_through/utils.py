import os
from collections.abc import Mapping
from types import MappingProxyType
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


_DECLARED_DEGRADATION_CHAINS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"max": ("max", "xhigh", "high"), "xhigh": ("xhigh", "high"), "minimal": ("minimal", "low")}
)


def _effort_from_declaration(model_info: ModelInfo, effort: str) -> str | None:
    """A declared level set is the WHOLE answer for this gate, so a level it omits degrades even
    where a per-level flag would have allowed it. Honoring both would let /model_group/info and
    this path disagree about the same entry. None means the entry declares nothing, and the flag
    chain below decides as before.

    A declaration that omits every level in a chain still lands on that chain's terminal, which can
    itself be undeclared. Picking a nearer declared level instead would need a strength ordering,
    and the advertisement order is presentation only by design, so the terminal stays the answer."""
    from litellm.router_utils.reasoning_effort_capability import declared_reasoning_efforts

    declared: Final = declared_reasoning_efforts(model_info)
    if declared is None:
        return None
    chain: Final = _DECLARED_DEGRADATION_CHAINS[effort]
    return next((level for level in chain if level in declared), chain[-1])


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

    declared_effort: Final = _effort_from_declaration(model_info, effort) if model_info is not None else None
    if declared_effort is not None:
        return declared_effort

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
