"""
Helper util for handling databricks-specific cost calculation
- e.g.: handling 'dbrx-instruct-*'
"""

from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage

_DATABRICKS_LEGACY_TO_CANONICAL: Final = MappingProxyType(
    {
        "databricks-claude-fable-5": "system.ai.claude-fable-5",
        "databricks-claude-fable-5-1": "system.ai.claude-fable-5-1",
        "databricks-claude-haiku-4-5": "system.ai.claude-haiku-4-5",
        "databricks-claude-opus-4-6": "system.ai.claude-opus-4-6",
        "databricks-claude-opus-4-7": "system.ai.claude-opus-4-7",
        "databricks-claude-opus-4-8": "system.ai.claude-opus-4-8",
        "databricks-claude-opus-5": "system.ai.claude-opus-5",
        "databricks-claude-sonnet-4": "system.ai.claude-sonnet-4",
        "databricks-claude-sonnet-4-5": "system.ai.claude-sonnet-4-5",
        "databricks-claude-sonnet-4-6": "system.ai.claude-sonnet-4-6",
        "databricks-claude-sonnet-5": "system.ai.claude-sonnet-5",
        "databricks-deepseek-v4-flash-0731": "system.ai.deepseek-v4-flash-0731",
        "databricks-deepseek-v4-pro-0813": "system.ai.deepseek-v4-pro-0813",
        "databricks-gemini-2-5-flash": "system.ai.gemini-2-5-flash",
        "databricks-gemini-2-5-pro": "system.ai.gemini-2-5-pro",
        "databricks-gemini-3-1-flash-lite": "system.ai.gemini-3-1-flash-lite",
        "databricks-gemini-3-1-pro": "system.ai.gemini-3-1-pro",
        "databricks-gemini-3-5-flash": "system.ai.gemini-3-5-flash",
        "databricks-gemini-3-6-flash": "system.ai.gemini-3-6-flash",
        "databricks-gemini-3-flash": "system.ai.gemini-3-flash",
        "databricks-gemma-3-12b": "system.ai.gemma-3-12b",
        "databricks-glm-5-2": "system.ai.glm-5-2",
        "databricks-glm-5-3": "system.ai.glm-5-3",
        "databricks-glm-5-3-flash": "system.ai.glm-5-3-flash",
        "databricks-gpt-5": "system.ai.gpt-5",
        "databricks-gpt-5-1": "system.ai.gpt-5-1",
        "databricks-gpt-5-2": "system.ai.gpt-5-2",
        "databricks-gpt-5-4": "system.ai.gpt-5-4",
        "databricks-gpt-5-4-mini": "system.ai.gpt-5-4-mini",
        "databricks-gpt-5-4-nano": "system.ai.gpt-5-4-nano",
        "databricks-gpt-5-5": "system.ai.gpt-5-5",
        "databricks-gpt-5-6-luna": "system.ai.gpt-5-6-luna",
        "databricks-gpt-5-6-sol": "system.ai.gpt-5-6-sol",
        "databricks-gpt-5-mini": "system.ai.gpt-5-mini",
        "databricks-gpt-5-nano": "system.ai.gpt-5-nano",
        "databricks-gpt-oss-120b": "system.ai.gpt-oss-120b",
        "databricks-gpt-oss-20b": "system.ai.gpt-oss-20b",
        "databricks-grok-4-6": "system.ai.grok-4-6",
        "databricks-gte-large-en": "system.ai.gte-large-en",
        "databricks-inkling": "system.ai.inkling",
        "databricks-kimi-k3": "system.ai.kimi-k3",
        "databricks-llama-4-maverick": "system.ai.llama-4-maverick",
        "databricks-meta-llama-3-1-8b-instruct": "system.ai.meta-llama-3-1-8b-instruct",
        "databricks-meta-llama-3-3-70b-instruct": "system.ai.meta-llama-3-3-70b-instruct",
        "databricks-qwen3-embedding-0-6b": "system.ai.qwen3-embedding-0-6b",
        "databricks-qwen3-next-80b-a3b-instruct": "system.ai.qwen3-next-80b-a3b-instruct",
        "databricks-qwen35-122b-a10b": "system.ai.qwen35-122b-a10b",
    }
)

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


def get_databricks_cost_key(model: str) -> str | None:
    name: Final = model.removeprefix("databricks/")
    canonical_slug: Final = _DATABRICKS_LEGACY_TO_CANONICAL.get(name)
    if canonical_slug is not None:
        return f"databricks/{canonical_slug}"
    legacy_endpoint: Final = _LEGACY_ENDPOINT_NAMES.get(name)
    if legacy_endpoint is not None:
        resolved_canonical: Final = _DATABRICKS_LEGACY_TO_CANONICAL.get(legacy_endpoint)
        if resolved_canonical is not None:
            return f"databricks/{resolved_canonical}"
        return f"databricks/{legacy_endpoint}"
    return None


def _registry_key(model: str) -> str:
    name: Final = model.removeprefix("databricks/")
    canonical_slug: Final = _DATABRICKS_LEGACY_TO_CANONICAL.get(name)
    if canonical_slug is not None:
        return canonical_slug
    endpoint: Final = next(
        (key for prefix, key in _LEGACY_ENDPOINT_NAMES.items() if name.startswith(prefix)),
        name,
    )
    return _DATABRICKS_LEGACY_TO_CANONICAL.get(endpoint, endpoint)


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return generic_cost_per_token(
        model=_registry_key(model),
        usage=usage,
        custom_llm_provider="databricks",
    )
