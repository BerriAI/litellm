"""
Helper util for handling databricks-specific cost calculation
- e.g.: handling 'dbrx-instruct-*'
"""

from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import branch_cost
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info

_LEGACY_ENDPOINT_NAMES: Final = MappingProxyType(
    {
        "dbrx-instruct": "databricks-dbrx-instruct",
        "meta-llama-3.1-70b-instruct": "databricks-meta-llama-3-1-70b-instruct",
        "meta-llama-3.1-405b-instruct": "databricks-meta-llama-3-1-405b-instruct",
        "mixtral-8x7b-instruct-v0.1": "databricks-mixtral-8x7b-instruct",
        "bge-large-en": "databricks-bge-large-en",
        "gte-large-en": "databricks-gte-large-en",
        "llama-2-70b-chat": "databricks-llama-2-70b-chat",
    }
)


def _registry_key(model: str) -> str:
    name: Final = model.removeprefix("databricks/")
    return next(
        (key for prefix, key in _LEGACY_ENDPOINT_NAMES.items() if name.startswith(prefix)),
        name,
    )


def pricing_row(model: str) -> ModelInfo:
    """The row a Databricks call prices with, legacy endpoint names under their ``databricks-`` key."""
    return get_model_info(model=_registry_key(model), custom_llm_provider="databricks")


def cost_per_token(model: str, usage: Usage, service_tier: str | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return branch_cost(_registry_key(model), "databricks", usage, pricing_row(model), service_tier=service_tier)
