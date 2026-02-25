"""
Returns a random deployment from the list of healthy deployments.

Simple shuffle without weighted pick - pure random selection.

"""

import random
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _get_deployment_info(deployment: Dict) -> str:
    """
    Helper to extract key deployment info for logging.

    Returns a string with deployment id, model, and api_base if available.
    """
    try:
        model_info = deployment.get("model_info", {})
        deployment_id = model_info.get("id", "unknown")

        litellm_params = deployment.get("litellm_params", {})
        model = litellm_params.get("model", "unknown")
        api_base = litellm_params.get("api_base", "unknown")

        return f"[id={deployment_id}, model={model}, api_base={api_base}]"
    except Exception as e:
        return f"[error extracting info: {e}]"


def simple_shuffle(
    llm_router_instance: LitellmRouter,
    healthy_deployments: Union[List[Any], Dict[Any, Any]],
    model: str,
) -> Dict:
    """
    Returns a random deployment from the list of healthy deployments.

    Simple shuffle without weighted pick - pure random selection.

    Args:
        llm_router_instance: LitellmRouter instance
        healthy_deployments: List of healthy deployments
        model: Model name

    Returns:
        Dict: A single healthy deployment
    """
    print(f"[Simple-Shuffle] Starting routing for model: {model}")

    # Handle edge cases
    if healthy_deployments is None or len(healthy_deployments) == 0:
        print(f"[Simple-Shuffle] WARNING: No healthy deployments available for model: {model}")
        raise ValueError(f"No healthy deployments available for model: {model}")

    if len(healthy_deployments) == 1:
        deployment = healthy_deployments[0]
        print(f"[Simple-Shuffle] Only one healthy deployment available for model: {model}, returning: {_get_deployment_info(deployment)}")
        return deployment

    # Log total number of healthy deployments
    print(f"[Simple-Shuffle] Total healthy deployments for model '{model}': {len(healthy_deployments)}")

    # Log details of all healthy deployments
    for idx, deployment in enumerate(healthy_deployments):
        print(f"[Simple-Shuffle] Healthy deployment {idx + 1}/{len(healthy_deployments)}: {_get_deployment_info(deployment)}")

    # Perform simple random pick
    selected_index = random.randrange(len(healthy_deployments))
    deployment = healthy_deployments[selected_index]

    print(f"[Simple-Shuffle] Selected deployment {selected_index + 1} of {len(healthy_deployments)}: {_get_deployment_info(deployment)} for model: {model}")

    return deployment
