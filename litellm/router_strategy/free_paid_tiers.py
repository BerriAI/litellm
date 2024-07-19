"""
Use this to route requests between free and paid tiers
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

from litellm._logging import verbose_logger
from litellm.types.router import DeploymentTypedDict


class ModelInfo(TypedDict):
    tier: Literal["free", "paid"]


class Deployment(TypedDict):
    model_info: ModelInfo


async def get_deployments_for_tier(
    request_kwargs: Optional[Dict[Any, Any]] = None,
    healthy_deployments: Optional[Union[List[Any], Dict[Any, Any]]] = None,
):
    """
    if request_kwargs contains {"metadata": {"tier": "free"}} or {"metadata": {"tier": "paid"}}, then routes the request to free/paid tier models
    """
    if request_kwargs is None:
        verbose_logger.debug(
            "get_deployments_for_tier: request_kwargs is None returning healthy_deployments: %s",
            healthy_deployments,
        )
        return healthy_deployments

    verbose_logger.debug("request metadata: %s", request_kwargs.get("metadata"))
    if "metadata" in request_kwargs:
        metadata = request_kwargs["metadata"]
        if "tier" in metadata:
            selected_tier: Literal["free", "paid"] = metadata["tier"]
            if healthy_deployments is None:
                return None

            if selected_tier == "free":
                # get all deployments where model_info has tier = free
                free_deployments: List[Any] = []
                verbose_logger.debug(
                    "Getting deployments in free tier, all_deployments: %s",
                    healthy_deployments,
                )
                for deployment in healthy_deployments:
                    typed_deployment = cast(Deployment, deployment)
                    if typed_deployment["model_info"]["tier"] == "free":
                        free_deployments.append(deployment)
                verbose_logger.debug("free_deployments: %s", free_deployments)
                return free_deployments

            elif selected_tier == "paid":
                # get all deployments where model_info has tier = paid
                paid_deployments: List[Any] = []
                for deployment in healthy_deployments:
                    typed_deployment = cast(Deployment, deployment)
                    if typed_deployment["model_info"]["tier"] == "paid":
                        paid_deployments.append(deployment)
                verbose_logger.debug("paid_deployments: %s", paid_deployments)
                return paid_deployments

    verbose_logger.debug(
        "no tier found in metadata, returning healthy_deployments: %s",
        healthy_deployments,
    )
    return healthy_deployments
