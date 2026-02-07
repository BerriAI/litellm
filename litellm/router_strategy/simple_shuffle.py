"""
Returns a random deployment from the list of healthy deployments.

If weights are provided, it will return a deployment based on the weights.

"""

import random
from typing import TYPE_CHECKING, Any, Dict, List, Union

from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def simple_shuffle(
    llm_router_instance: LitellmRouter,
    healthy_deployments: Union[List[Any], Dict[Any, Any]],
    model: str,
) -> Dict:
    """
    Returns a random deployment from the list of healthy deployments.

    If weights are provided, it will return a deployment based on the weights.

    If users pass `rpm` or `tpm`, we do a random weighted pick - based on `rpm`/`tpm`.

    Args:
        llm_router_instance: LitellmRouter instance
        healthy_deployments: List of healthy deployments
        model: Model name

    Returns:
        Dict: A single healthy deployment
    """

    ############## Check if 'weight' or 'rpm' or 'tpm' param set for a weighted pick #################
    for weight_by in ["weight", "rpm", "tpm"]:
        weight = healthy_deployments[0].get("litellm_params").get(weight_by, None)
        if weight is not None:
            weights = [m["litellm_params"].get(weight_by, 0) for m in healthy_deployments]
            verbose_router_logger.debug(f"\nweight {weights}")
            total_weight = sum(weights)
            weights = [weight / total_weight for weight in weights]
            verbose_router_logger.debug(f"\n weights {weights} by {weight_by}")
            # Perform weighted random pick
            selected_index = random.choices(range(len(weights)), weights=weights)[0]
            verbose_router_logger.debug(f"\n selected index, {selected_index}")
            deployment = healthy_deployments[selected_index]
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, Selected deployment: {llm_router_instance.print_deployment(deployment) or deployment[0]} for model: {model}"
            )
            return deployment or deployment[0]


    ############## No RPM/TPM passed, we do a random pick #################
    item = random.choice(healthy_deployments)
    return item or item[0]
