"""
Handles calculating cost for together ai models
"""

import re

from litellm.types.utils import CallTypes


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_model_params_and_category(model_name, call_type: CallTypes) -> str:
    """
    Helper function for calculating together ai pricing.

    Returns
    - str - model pricing category if mapped else received model name
    """
    if call_type == CallTypes.embedding or call_type == CallTypes.aembedding:
        return get_model_params_and_category_embeddings(model_name=model_name)
    model_name = model_name.lower()
    re_params_match = re.search(
        r"(\d+b)", model_name
    )  # catch all decimals like 3b, 70b, etc
    category = None
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_match = params_match.replace("b", "")
        if params_match is not None:
            params_billion = float(params_match)
        else:
            return model_name
        # Determine the category based on the number of parameters
        if params_billion <= 4.0:
            category = "together-ai-up-to-4b"
        elif params_billion <= 8.0:
            category = "together-ai-4.1b-8b"
        elif params_billion <= 21.0:
            category = "together-ai-8.1b-21b"
        elif params_billion <= 41.0:
            category = "together-ai-21.1b-41b"
        elif params_billion <= 80.0:
            category = "together-ai-41.1b-80b"
        elif params_billion <= 110.0:
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
    re_params_match = re.search(
        r"(\d+m)", model_name
    )  # catch all decimals like 100m, 200m, etc.
    category = None
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_match = params_match.replace("m", "")
        if params_match is not None:
            params_million = float(params_match)
        else:
            return model_name
        # Determine the category based on the number of parameters
        if params_million <= 150:
            category = "together-ai-embedding-up-to-150m"
        elif params_million <= 350:
            category = "together-ai-embedding-151m-to-350m"
        if category is not None:
            return category

    return model_name
