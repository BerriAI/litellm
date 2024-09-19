# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import logging
import random
from typing import List, Optional

import litellm
from litellm._logging import print_verbose

logger = logging.getLogger(__name__)


ILLEGAL_DISPLAY_PARAMS = [
    "messages",
    "api_key",
    "prompt",
    "input",
    "vertex_credentials",
    "aws_access_key_id",
    "aws_secret_access_key",
]

MINIMAL_DISPLAY_PARAMS = ["model", "mode_error"]


def _get_random_llm_message():
    """
    Get a random message from the LLM.
    """
    messages = ["Hey how's it going?", "What's 1 + 1?"]

    return [{"role": "user", "content": random.choice(messages)}]


def _clean_endpoint_data(endpoint_data: dict, details: Optional[bool] = True):
    """
    Clean the endpoint data for display to users.
    """
    return (
        {k: v for k, v in endpoint_data.items() if k not in ILLEGAL_DISPLAY_PARAMS}
        if details is not False
        else {k: v for k, v in endpoint_data.items() if k in MINIMAL_DISPLAY_PARAMS}
    )


def filter_deployments_by_id(
    model_list: List,
) -> List:
    seen_ids = set()
    filtered_deployments = []

    for deployment in model_list:
        _model_info = deployment.get("model_info") or {}
        _id = _model_info.get("id") or None
        if _id is None:
            continue

        if _id not in seen_ids:
            seen_ids.add(_id)
            filtered_deployments.append(deployment)

    return filtered_deployments


async def _perform_health_check(model_list: list, details: Optional[bool] = True):
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
        litellm_params = model["litellm_params"]

        if isinstance(is_healthy, dict) and "error" not in is_healthy:
            healthy_endpoints.append(
                _clean_endpoint_data({**litellm_params, **is_healthy}, details)
            )
        elif isinstance(is_healthy, dict):
            unhealthy_endpoints.append(
                _clean_endpoint_data({**litellm_params, **is_healthy}, details)
            )
        else:
            unhealthy_endpoints.append(_clean_endpoint_data(litellm_params, details))

    return healthy_endpoints, unhealthy_endpoints


async def perform_health_check(
    model_list: list,
    model: Optional[str] = None,
    cli_model: Optional[str] = None,
    details: Optional[bool] = True,
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

    model_list = filter_deployments_by_id(
        model_list=model_list
    )  # filter duplicate deployments (e.g. when model alias'es are used)
    healthy_endpoints, unhealthy_endpoints = await _perform_health_check(
        model_list, details
    )

    return healthy_endpoints, unhealthy_endpoints
