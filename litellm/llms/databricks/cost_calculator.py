"""
Helper util for handling databricks-specific cost calculation
- e.g.: handling 'dbrx-instruct-*'

Token billing (cache read/write, audio, reasoning) is delegated to
generic_cost_per_token so Databricks stays consistent with other
OpenAI-compatible providers.
"""

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def _resolve_databricks_base_model(model: str) -> str:
    """
    Map common Databricks deployment aliases onto pricing-table keys.

    The hand-rolled arithmetic previously lived next to this remapping; keep
    the remapping so callers that still pass bare foundation-model ids resolve
    the same entries as before.
    """
    if model.startswith("databricks/dbrx-instruct") or model.startswith("dbrx-instruct"):
        return "databricks-dbrx-instruct"
    if model.startswith("databricks/meta-llama-3.1-70b-instruct") or model.startswith("meta-llama-3.1-70b-instruct"):
        return "databricks-meta-llama-3-1-70b-instruct"
    if model.startswith("databricks/meta-llama-3.1-405b-instruct") or model.startswith("meta-llama-3.1-405b-instruct"):
        return "databricks-meta-llama-3-1-405b-instruct"
    if (
        model.startswith("databricks/mixtral-8x7b-instruct-v0.1")
        or model.startswith("mixtral-8x7b-instruct-v0.1")
        or model.startswith("databricks/mixtral-8x7b-instruct-v0.1")
        or model.startswith("mixtral-8x7b-instruct-v0.1")
    ):
        return "databricks-mixtral-8x7b-instruct"
    if model.startswith("databricks/bge-large-en") or model.startswith("bge-large-en"):
        return "databricks-bge-large-en"
    if model.startswith("databricks/gte-large-en") or model.startswith("gte-large-en"):
        return "databricks-gte-large-en"
    if model.startswith("databricks/llama-2-70b-chat") or model.startswith("llama-2-70b-chat"):
        return "databricks-llama-2-70b-chat"
    return model


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    base_model = _resolve_databricks_base_model(model)
    return generic_cost_per_token(model=base_model, usage=usage, custom_llm_provider="databricks")
