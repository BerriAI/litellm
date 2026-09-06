import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import litellm
from litellm.types.utils import ModelInfo

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject

OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH: Final = 64

# Weakest to strongest. ``none`` is deliberately absent: it is an off switch rather than a tier, so
# leaving it out of the ladder is what keeps it from ever being chosen for a caller who asked to think.
_EFFORT_STRENGTH_ORDER: Final[tuple[str, ...]] = ("minimal", "low", "medium", "high", "xhigh", "max")

# Where a tier lands when the deployment's accepted set cannot be read at all. Only the tiers that
# shipped with a floor keep one; every other tier keeps the level the caller asked for.
_EFFORT_BLIND_FALLBACK: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max": "high",
        "xhigh": "high",
        "minimal": "low",
    }
)


def _degradation_chain(effort: str) -> tuple[str, ...]:
    """Nearest-first search order for one ask: the level itself, then weaker tiers, then stronger.

    Lowering is preferred to raising because a tier the deployment does not accept is a hard 400,
    while a weaker one costs thinking budget rather than the request. The floor of the ladder has
    nothing weaker to fall to, so ``minimal`` climbs, which is the order it shipped with.
    """
    index: Final = _EFFORT_STRENGTH_ORDER.index(effort)
    return (*_EFFORT_STRENGTH_ORDER[index::-1], *_EFFORT_STRENGTH_ORDER[index + 1 :])


def prompt_cache_key_from_user_id(user_id: object) -> str | None:
    if user_id is None:
        return None
    return str(user_id)[:OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH] or None


def litellm_logging_obj_from_kwargs(kwargs: Mapping[str, object]) -> "LiteLLMLoggingObject | None":
    """The logging object the bridged call logs through, when the caller supplied one."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    candidate: Final = kwargs.get("litellm_logging_obj")
    return candidate if isinstance(candidate, Logging) else None


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
    """Resolve a tier against the levels the deployment accepts, to the nearest one it does.

    The accepted set is resolved by the same owner that answers ``/model_group/info``, so a level
    the proxy advertises is a level this path forwards.

    Every tier is resolved, not just the ones with an opt-in flag. An entry naming its levels
    outright can omit ``high``, ``medium`` or ``low``, and a level an entry omits is a level it
    rejects, so the tier the caller asked for cannot be what decides whether the declaration is
    read. ``none`` is never degraded to and is never chosen, being an off switch rather than a
    tier; an always-on-thinking model is handled where the thinking block is built.

    A deployment the map cannot answer for keeps the historical floor on the three tiers that
    shipped with one and the caller's own level on the rest, since there is nothing to resolve
    against and lowering blind would weaken deployments that never asked for it.
    """
    if effort not in _EFFORT_STRENGTH_ORDER:
        return effort

    from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts
    from litellm.utils import get_model_info

    blind_fallback: Final = _EFFORT_BLIND_FALLBACK.get(effort, effort)
    try:
        model_info: Final[ModelInfo] = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return blind_fallback

    supported: Final = resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True)
    if not supported:
        return blind_fallback

    return next((level for level in _degradation_chain(effort) if level in supported), blind_fallback)
