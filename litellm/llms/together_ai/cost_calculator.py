"""
Handles calculating cost for together ai models
"""

import re
from typing import Final

import litellm
from litellm.constants import (
    TOGETHER_AI_4_B,
    TOGETHER_AI_8_B,
    TOGETHER_AI_21_B,
    TOGETHER_AI_41_B,
    TOGETHER_AI_80_B,
    TOGETHER_AI_110_B,
    TOGETHER_AI_EMBEDDING_150_M,
    TOGETHER_AI_EMBEDDING_350_M,
)
from litellm.types.utils import CallTypes


def has_published_rate(model_name: str) -> bool:
    """Whether the cost map prices this model directly.

    The size buckets below are a fallback for ids the map has never priced. Applied
    unconditionally they shadow every entry whose name carries a parameter count, so a
    model with a published per-token rate (and a published cached-input rate) would bill
    at its bucket instead.
    """
    from litellm.utils import _get_model_cost_key

    bare: Final = model_name.removeprefix("together_ai/")
    for candidate in (f"together_ai/{bare}", model_name):
        resolved = _get_model_cost_key(candidate)
        if resolved is None:
            continue
        if litellm.model_cost[resolved].get("input_cost_per_token") is not None:
            return True
    return False


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_model_params_and_category(model_name, call_type: CallTypes) -> str:
    """
    Helper function for calculating together ai pricing.

    Returns
    - str - model pricing category if mapped else received model name
    """
    if has_published_rate(model_name):
        return model_name
    if call_type == CallTypes.embedding or call_type == CallTypes.aembedding:
        return get_model_params_and_category_embeddings(model_name=model_name)
    model_name = model_name.lower()
    re_params_match: Final = re.search(r"(\d+b)", model_name)  # catch all decimals like 3b, 70b, etc
    category = None
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_match = params_match.replace("b", "")
        if params_match is not None:
            params_billion: Final = float(params_match)
        else:
            return model_name
        # Determine the category based on the number of parameters
        if params_billion <= TOGETHER_AI_4_B:
            category = "together-ai-up-to-4b"
        elif params_billion <= TOGETHER_AI_8_B:
            category = "together-ai-4.1b-8b"
        elif params_billion <= TOGETHER_AI_21_B:
            category = "together-ai-8.1b-21b"
        elif params_billion <= TOGETHER_AI_41_B:
            category = "together-ai-21.1b-41b"
        elif params_billion <= TOGETHER_AI_80_B:
            category = "together-ai-41.1b-80b"
        elif params_billion <= TOGETHER_AI_110_B:
            category = "together-ai-81.1b-110b"
        if category is not None:
            return category

    return model_name


def get_model_params_and_category_embeddings(model_name) -> str:
    """
    Helper function for calculating together ai embedding pricing.

    Returns
    - str - model pricing category if mapped else received model name
    """
    model_name = model_name.lower()
    re_params_match: Final = re.search(r"(\d+m)", model_name)  # catch all decimals like 100m, 200m, etc.
    category = None
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_match = params_match.replace("m", "")
        if params_match is not None:
            params_million: Final = float(params_match)
        else:
            return model_name
        # Determine the category based on the number of parameters
        if params_million <= TOGETHER_AI_EMBEDDING_150_M:
            category = "together-ai-embedding-up-to-150m"
        elif params_million <= TOGETHER_AI_EMBEDDING_350_M:
            category = "together-ai-embedding-151m-to-350m"
        if category is not None:
            return category

    return model_name
