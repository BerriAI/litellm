# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import random
from typing import Optional

import litellm
import logging
from litellm._logging import print_verbose


logger = logging.getLogger(__name__)


ILLEGAL_DISPLAY_PARAMS = [
    "messages",
    "api_key"
]


def _get_random_llm_message():
    """
    Get a random message from the LLM.
    """
    messages = [
        "Hey how's it going?",
        "What's 1 + 1?"
    ]


    return [
        {"role": "user", "content": random.choice(messages)}
    ]


def _clean_litellm_params(litellm_params: dict):
    """
    Clean the litellm params for display to users.
    """
    return {k: v for k, v in litellm_params.items() if k not in ILLEGAL_DISPLAY_PARAMS}


async def _perform_health_check(model_list: list):
    """
    Perform a health check for each model in the list.
    """
    async def _check_embedding_model(model_params: dict):
        model_params.pop("messages", None)
        model_params["input"] = ["test from litellm"]
        try:
            await litellm.aembedding(**model_params)
        except Exception as e:
            print_verbose(f"\n\n Got Exception, {e}")
            print_verbose(f"Health check failed for model {model_params['model']}. Error: {e}")
            return False
        return True


    async def _check_model(model_params: dict):
        try:
            await litellm.acompletion(**model_params)
        except Exception as e:
            print_verbose(f"\n\n Got Exception, {e}")
            error_str = (str(e))
            if "This is not a chat model" in error_str or "The chatCompletion operation does not work with the specified model" in error_str:
                    return await _check_embedding_model(model_params)
                
            print_verbose(f"Health check failed for model {model_params['model']}. Error: {e}")
            return False
        
        return True

    prepped_params = []
    tasks = []
    for model in model_list:
        litellm_params = model["litellm_params"]
        litellm_params["model"] = litellm.utils.remove_model_id(litellm_params["model"])
        litellm_params["messages"] = _get_random_llm_message()

        prepped_params.append(litellm_params)
        if "embedding" in litellm_params["model"]:
            tasks.append(_check_embedding_model(litellm_params))
        else:
            tasks.append(_check_model(litellm_params))

    results = await asyncio.gather(*tasks)

    healthy_endpoints = []
    unhealthy_endpoints = []

    for is_healthy, model in zip(results, model_list):
        cleaned_litellm_params = _clean_litellm_params(model["litellm_params"])

        if is_healthy:
            healthy_endpoints.append(cleaned_litellm_params)
        else:
            unhealthy_endpoints.append(cleaned_litellm_params)

    return healthy_endpoints, unhealthy_endpoints




async def perform_health_check(model_list: list, model: Optional[str] = None):
    """
    Perform a health check on the system.

    Returns:
        (bool): True if the health check passes, False otherwise.
    """
    if not model_list:
        return [], []

    if model is not None:
        model_list = [x for x in model_list if x["litellm_params"]["model"] == model]

    healthy_endpoints, unhealthy_endpoints = await _perform_health_check(model_list)

    return healthy_endpoints, unhealthy_endpoints

    