"""
Use this to route requests between Teams

- If tags in request is a subset of tags in deployment, return deployment
- if deployments are set with default tags, return all default deployment
- If no default_deployments are set, return all deployments
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.router import RouterErrors

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


async def get_deployments_for_tag(
    llm_router_instance: LitellmRouter,
    model: str,  # used to raise the correct error
    healthy_deployments: Union[List[Any], Dict[Any, Any]],
    request_kwargs: Optional[Dict[Any, Any]] = None,
):
    """
    Returns a list of deployments that match the requested model and tags in the request.

    Executes tag based filtering based on the tags in request metadata and the tags on the deployments
    """
    if llm_router_instance.enable_tag_filtering is not True:
        return healthy_deployments

    if request_kwargs is None:
        verbose_logger.debug(
            "get_deployments_for_tag: request_kwargs is None returning healthy_deployments: %s",
            healthy_deployments,
        )
        return healthy_deployments

    if healthy_deployments is None:
        verbose_logger.debug(
            "get_deployments_for_tag: healthy_deployments is None returning healthy_deployments"
        )
        return healthy_deployments

    verbose_logger.debug("request metadata: %s", request_kwargs.get("metadata"))
    if "metadata" in request_kwargs:
        metadata = request_kwargs["metadata"]
        request_tags = metadata.get("tags")

        new_healthy_deployments = []
        if request_tags:
            verbose_logger.debug(
                "get_deployments_for_tag routing: router_keys: %s", request_tags
            )
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
                elif "default" in deployment_tags:
                    verbose_logger.debug(
                        "adding default deployment with tags: %s, request tags: %s",
                        deployment_tags,
                        request_tags,
                    )
                    new_healthy_deployments.append(deployment)

            if len(new_healthy_deployments) == 0:
                raise ValueError(
                    f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model={model} and tags={request_tags}"
                )

            return new_healthy_deployments

    # for Untagged requests use default deployments if set
    _default_deployments_with_tags = []
    for deployment in healthy_deployments:
        if "default" in deployment.get("litellm_params", {}).get("tags", []):
            _default_deployments_with_tags.append(deployment)

    if len(_default_deployments_with_tags) > 0:
        return _default_deployments_with_tags

    # if no default deployment is found, return healthy_deployments
    verbose_logger.debug(
        "no tier found in metadata, returning healthy_deployments: %s",
        healthy_deployments,
    )
    return healthy_deployments


def _get_tags_from_request_kwargs(
    request_kwargs: Optional[Dict[Any, Any]] = None
) -> List[str]:
    """
    Helper to get tags from request kwargs

    Args:
        request_kwargs: The request kwargs to get tags from

    Returns:
        List[str]: The tags from the request kwargs
    """
    if request_kwargs is None:
        return []
    if "metadata" in request_kwargs:
        metadata = request_kwargs["metadata"]
        return metadata.get("tags", [])
    elif "litellm_params" in request_kwargs:
        litellm_params = request_kwargs["litellm_params"]
        _metadata = litellm_params.get("metadata", {})
        return _metadata.get("tags", [])
    return []
