# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import logging
import random
from typing import List, Optional

import litellm

logger = logging.getLogger(__name__)
from litellm.constants import HEALTH_CHECK_TIMEOUT_SECONDS

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
    endpoint_data.pop("litellm_logging_obj", None)
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


async def run_with_timeout(task, timeout):
    try:
        return await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError:
        task.cancel()
        # Only cancel child tasks of the current task
        current_task = asyncio.current_task()
        for t in asyncio.all_tasks():
            if t != current_task:
                t.cancel()
        try:
            await asyncio.wait_for(task, 0.1)  # Give 100ms for cleanup
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        return {"error": "Timeout exceeded"}


async def _perform_health_check(model_list: list, details: Optional[bool] = True):
    """
    Perform a health check for each model in the list.
    """

    tasks = []
    for model in model_list:
        litellm_params = model["litellm_params"]
        model_info = model.get("model_info", {})
        mode = model_info.get("mode", None)
        litellm_params = _update_litellm_params_for_health_check(
            model_info, litellm_params
        )
        timeout = model_info.get("health_check_timeout") or HEALTH_CHECK_TIMEOUT_SECONDS

        task = run_with_timeout(
            litellm.ahealth_check(
                model["litellm_params"],
                mode=mode,
                prompt="test from litellm",
                input=["test from litellm"],
            ),
            timeout,
        )

        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

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


def _update_litellm_params_for_health_check(
    model_info: dict, litellm_params: dict
) -> dict:
    """
    Update the litellm params for health check.

    - gets a short `messages` param for health check
    - updates the `model` param with the `health_check_model` if it exists Doc: https://docs.litellm.ai/docs/proxy/health#wildcard-routes
    - updates the `voice` param with the `health_check_voice` for `audio_speech` mode if it exists Doc: https://docs.litellm.ai/docs/proxy/health#text-to-speech-models
    - updates the `model` param with the Bedrock base model name if it is a Bedrock model
    """
    litellm_params["messages"] = _get_random_llm_message()
    _health_check_model = model_info.get("health_check_model", None)
    if _health_check_model is not None:
        litellm_params["model"] = _health_check_model
    if model_info.get("mode", None) == "audio_speech":
        litellm_params["voice"] = model_info.get("health_check_voice", "alloy")
    if "bedrock" in litellm_params["model"]:
        from litellm.llms.bedrock.common_utils import BedrockModelInfo
        litellm_params["model"] = BedrockModelInfo.get_base_model(litellm_params["model"])
    return litellm_params


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
