from typing import TYPE_CHECKING

from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT

if TYPE_CHECKING:
    from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig


def resolve_warm_models(config: "ComplexityRouterConfig") -> tuple[str, ...]:
    explicit = config.cache_warming.warm_models
    if explicit:
        return tuple(dict.fromkeys(explicit))
    first_per_tier = (models if isinstance(models, str) else models[0] for models in config.tiers.values() if models)
    return tuple(dict.fromkeys(first_per_tier))


def min_prompt_cache_tokens_for_warm_set(warm_models: tuple[str, ...]) -> int:
    from litellm.utils import get_prompt_cache_min_tokens

    if not warm_models:
        return DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
    return min(get_prompt_cache_min_tokens(model) for model in warm_models)
