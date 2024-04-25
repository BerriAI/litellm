# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import random
from typing import Optional

import litellm
import logging
from litellm._logging import print_verbose


logger = logging.getLogger(__name__)


ILLEGAL_DISPLAY_PARAMS = ["messages", "api_key", "prompt", "input"]


def _get_random_llm_message():
    """
    Get a random message from the LLM.
    """
    messages = ["Hey how's it going?", "What's 1 + 1?"]

    return [{"role": "user", "content": random.choice(messages)}]


def _clean_litellm_params(litellm_params: dict):
    """
    Clean the litellm params for display to users.
    """
    return {k: v for k, v in litellm_params.items() if k not in ILLEGAL_DISPLAY_PARAMS}


async def _perform_health_check(model_list: list):
    """
    Perform a health check for each model in the list.
    """
    tasks = []
    for model in model_list:
        litellm_params = model["litellm_params"]
        model_info = model.get("model_info", {})
        litellm_params["messages"] = _get_random_llm_message()
        mode = model_info.get("mode", None)
        tasks.append(
            litellm.ahealth_check(
                litellm_params,
                mode=mode,
                prompt="test from litellm",
                input=["test from litellm"],
            )
        )

    results = await asyncio.gather(*tasks)

    healthy_endpoints = []
    unhealthy_endpoints = []

    for is_healthy, model in zip(results, model_list):
        cleaned_litellm_params = _clean_litellm_params(model["litellm_params"])

        if isinstance(is_healthy, dict) and "error" not in is_healthy:
            healthy_endpoints.append({**cleaned_litellm_params, **is_healthy})
        elif isinstance(is_healthy, dict):
            unhealthy_endpoints.append({**cleaned_litellm_params, **is_healthy})
        else:
            unhealthy_endpoints.append(cleaned_litellm_params)

    return healthy_endpoints, unhealthy_endpoints


async def perform_health_check(
    model_list: list, model: Optional[str] = None, cli_model: Optional[str] = None
):
    """
    Perform a health check on the system.

    Returns:
        (bool): True if the health check passes, False otherwise.
    """
    if not model_list:
        if cli_model:
            model_list = [
                {"model_name": cli_model, "litellm_params": {"model": cli_model}}
            ]
        else:
            return [], []

    if model is not None:
        _new_model_list = [
            x for x in model_list if x["litellm_params"]["model"] == model
        ]
        if _new_model_list == []:
            _new_model_list = [x for x in model_list if x["model_name"] == model]
        model_list = _new_model_list

    healthy_endpoints, unhealthy_endpoints = await _perform_health_check(model_list)

    return healthy_endpoints, unhealthy_endpoints
