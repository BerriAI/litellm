"""
Returns a random deployment from the list of healthy deployments.

If weights are provided, it will return a deployment based on the weights.

"""

import random
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def simple_shuffle(
    llm_router_instance: LitellmRouter,
    healthy_deployments: list[Any] | dict[Any, Any],
    model: str,
) -> dict:
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
            verbose_router_logger.debug("\nweight %s", weights)
            total_weight = sum(weights)
            if total_weight <= 0:
                # All remaining candidates have weight 0 for this metric (e.g.
                # after a weighted-failover exclusion left only zero-weight
                # backups). Skip to the next metric (rpm/tpm) which may still
                # provide a meaningful weighted pick; if none do, we fall
                # through to the uniform random pick at the end.
                continue
            weights = [weight / total_weight for weight in weights]
            verbose_router_logger.debug("\n weights %s by %s", weights, weight_by)
            # Perform weighted random pick
            selected_index = random.choices(range(len(weights)), weights=weights)[0]
            verbose_router_logger.debug("\n selected index, %s", selected_index)
            deployment = healthy_deployments[selected_index]
            verbose_router_logger.info(
                "get_available_deployment for model: %s, Selected deployment: %s for model: %s",
                model,
                llm_router_instance.print_deployment(deployment) or deployment[0],
                model,
            )
            return deployment or deployment[0]

    ############## No RPM/TPM passed, we do a random pick #################
    item: Final = random.choice(healthy_deployments)
    return item or item[0]
