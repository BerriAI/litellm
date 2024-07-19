"""
Use this to route requests between free and paid tiers
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

from litellm._logging import verbose_logger
from litellm.types.router import DeploymentTypedDict


async def get_deployments_for_tag(
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

    if healthy_deployments is None:
        verbose_logger.debug(
            "get_deployments_for_tier: healthy_deployments is None returning healthy_deployments"
        )
        return healthy_deployments

    verbose_logger.debug("request metadata: %s", request_kwargs.get("metadata"))
    if "metadata" in request_kwargs:
        metadata = request_kwargs["metadata"]
        request_tags = metadata.get("tags")

        new_healthy_deployments = []
        if request_tags:
            verbose_logger.debug("parameter routing: router_keys: %s", request_tags)
            # example this can be router_keys=["free", "custom"]
            # get all deployments that have a superset of these router keys
            for deployment in healthy_deployments:
                deployment_litellm_params = deployment.get("litellm_params")
                deployment_tags = deployment_litellm_params.get("tags")

                verbose_logger.debug(
                    "deployment: %s,  deployment_router_keys: %s",
                    deployment,
                    deployment_tags,
                )

                if deployment_tags is None:
                    continue

                if set(request_tags).issubset(set(deployment_tags)):
                    verbose_logger.debug(
                        "adding deployment with tags: %s, request tags: %s",
                        deployment_tags,
                        request_tags,
                    )
                    new_healthy_deployments.append(deployment)

        return new_healthy_deployments

    verbose_logger.debug(
        "no tier found in metadata, returning healthy_deployments: %s",
        healthy_deployments,
    )
    return healthy_deployments
