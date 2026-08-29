import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import litellm
from litellm.types.utils import ModelInfo

OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH: Final = 64

_EFFORT_DEGRADATION_CHAIN: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "max": ("max", "xhigh", "high"),
        "xhigh": ("xhigh", "high"),
        "minimal": ("minimal", "low"),
    }
)
_THINKING_OFF: Final = "none"


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
    """Lower a tier the deployment does not accept to the nearest one it does, leaving others alone.

    The accepted set is resolved by the same owner that answers ``/model_group/info``, so a level
    the proxy advertises is a level this path forwards.

    A deployment that refuses every step of a chain falls back to an accepted level read off that
    same set rather than to an assumed one, since an entry naming its levels outright can exclude
    the tiers the per-level flags treat as unconditional. ``none`` is never that fallback and is
    never degraded to, being an off switch rather than a tier; an always-on-thinking model is
    handled where the thinking block is built. A deployment accepting no tier at all keeps the
    chain's floor, which is what every deployment degraded to before there was anything to ask.
    """
    chain: Final = _EFFORT_DEGRADATION_CHAIN.get(effort)
    if chain is None:
        return effort

    from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts
    from litellm.utils import get_model_info

    try:
        model_info: Final[ModelInfo] = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return chain[-1]

    supported: Final = resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True)
    if not supported:
        return chain[-1]

    accepted_tiers: Final = tuple(level for level in supported if level != _THINKING_OFF)
    return next((level for level in (*chain, *accepted_tiers) if level in supported), chain[-1])
