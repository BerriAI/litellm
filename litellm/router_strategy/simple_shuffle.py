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
    # Handle edge cases
    if healthy_deployments is None or len(healthy_deployments) == 0:
        raise ValueError(f"No healthy deployments available for model: {model}")

    if len(healthy_deployments) == 1:
        return healthy_deployments[0]

    # Perform simple random pick
    selected_index = random.randrange(len(healthy_deployments))
    return healthy_deployments[selected_index]
