from typing import TYPE_CHECKING

from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT

if TYPE_CHECKING:
    from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig


def resolve_warm_models(config: "ComplexityRouterConfig") -> tuple[str, ...]:
    """Every model a session on this auto-router could end up being served by, which is the universe warming
    draws from rather than the set it warms. A session is warmed only on the members it has actually been
    served on, so this is used to bound eligibility and the capture-time minimum-token gate. An explicit
    warm_models narrows the universe rather than widening it: warming never replays against a model the
    session has not used, so the operator setting is an allowlist over the pool, not a pre-warm list."""
    explicit = config.cache_warming.warm_models
    if explicit:
        return tuple(dict.fromkeys(explicit))
    pooled = (models if isinstance(models, list) else [models] for models in config.tiers.values() if models)
    return tuple(dict.fromkeys(model for pool in pooled for model in pool))


def min_prompt_cache_tokens_for_warm_set(warm_models: tuple[str, ...]) -> int:
    from litellm.utils import get_prompt_cache_min_tokens

    if not warm_models:
        return DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
    return min(get_prompt_cache_min_tokens(model) for model in warm_models)
