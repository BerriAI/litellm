"""
Helper util for handling databricks-specific cost calculation
- e.g.: handling 'dbrx-instruct-*'
"""

from typing import Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    InputCostBreakdown,
    OutputCostBreakdown,
)
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(
    model: str, usage: Usage
) -> Tuple[InputCostBreakdown, OutputCostBreakdown]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block

    Returns:
        Tuple[InputCostBreakdown, OutputCostBreakdown] - granular input and output cost breakdowns
    """
    base_model = model
    if model.startswith("databricks/dbrx-instruct") or model.startswith(
        "dbrx-instruct"
    ):
        base_model = "databricks-dbrx-instruct"
    elif model.startswith("databricks/meta-llama-3.1-70b-instruct") or model.startswith(
        "meta-llama-3.1-70b-instruct"
    ):
        base_model = "databricks-meta-llama-3-1-70b-instruct"
    elif model.startswith(
        "databricks/meta-llama-3.1-405b-instruct"
    ) or model.startswith("meta-llama-3.1-405b-instruct"):
        base_model = "databricks-meta-llama-3-1-405b-instruct"
    elif model.startswith("databricks/mixtral-8x7b-instruct-v0.1") or model.startswith(
        "mixtral-8x7b-instruct-v0.1"
    ):
        base_model = "databricks-mixtral-8x7b-instruct"
    elif model.startswith("databricks/mixtral-8x7b-instruct-v0.1") or model.startswith(
        "mixtral-8x7b-instruct-v0.1"
    ):
        base_model = "databricks-mixtral-8x7b-instruct"
    elif model.startswith("databricks/bge-large-en") or model.startswith(
        "bge-large-en"
    ):
        base_model = "databricks-bge-large-en"
    elif model.startswith("databricks/gte-large-en") or model.startswith(
        "gte-large-en"
    ):
        base_model = "databricks-gte-large-en"
    elif model.startswith("databricks/llama-2-70b-chat") or model.startswith(
        "llama-2-70b-chat"
    ):
        base_model = "databricks-llama-2-70b-chat"

    model_info = get_model_info(model=base_model, custom_llm_provider="databricks")

    prompt_cost: float = usage["prompt_tokens"] * model_info["input_cost_per_token"]
    completion_cost: float = (
        usage["completion_tokens"] * model_info["output_cost_per_token"]
    )

    return (
        InputCostBreakdown(total=prompt_cost, text_cost=prompt_cost),
        OutputCostBreakdown(total=completion_cost, text_cost=completion_cost),
    )
