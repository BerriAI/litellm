"""
Use this to route requests between Teams

- If tags in request is a subset of tags in deployment, return deployment
- if deployments are set with default tags, return all default deployment
- If no default_deployments are set, return all deployments
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.router import RouterErrors

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def is_valid_deployment_tag(
    deployment_tags: List[str], request_tags: List[str], match_any: bool = True
) -> bool:
    """
    Check if a tag is valid, the matching can be either any or all based on `match_any` flag
    """
    if not request_tags:
        return False

    dep_set = set(deployment_tags)
    req_set = set(request_tags)

    if match_any:
        is_valid_deployment = bool(dep_set & req_set)
    else:
        is_valid_deployment = req_set.issubset(dep_set)

    if is_valid_deployment:
        verbose_logger.debug(
            "adding deployment with tags: %s, request tags: %s for match_any=%s",
            deployment_tags,
            request_tags,
            match_any,
        )
        return True
    return False


async def get_deployments_for_tag(
    llm_router_instance: LitellmRouter,
    model: str,  # used to raise the correct error
    healthy_deployments: Union[List[Any], Dict[Any, Any]],
    request_kwargs: Optional[Dict[Any, Any]] = None,
    metadata_variable_name: Literal["metadata", "litellm_metadata"] = "metadata",
):
    """
    Returns a list of deployments that match the requested model and tags in the request.

    Executes tag based filtering based on the tags in request metadata and the tags on the deployments
    """
    # Three-state behavior for enable_tag_filtering:
    #   False  → opt-out: never apply tag filtering
    #   None   → auto: apply when request carries tags, graceful fallthrough on no match
    #   True   → strict: apply always, ValueError on no match
    if llm_router_instance.enable_tag_filtering is False:
        return healthy_deployments

    request_tags = _get_tags_from_request_kwargs(
        request_kwargs, metadata_variable_name
    )
    if not request_tags and llm_router_instance.enable_tag_filtering is not True:
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

    verbose_logger.debug(
        "request metadata: %s", request_kwargs.get(metadata_variable_name)
    )
    if metadata_variable_name in request_kwargs:
        # reuse request_tags extracted by _get_tags_from_request_kwargs above
        match_any = llm_router_instance.tag_filtering_match_any

        new_healthy_deployments = []
        default_deployments = []
        if request_tags:
            if llm_router_instance.enable_tag_filtering is not True:
                if not getattr(llm_router_instance, "_tag_filtering_auto_logged", False):
                    verbose_logger.info(
                        "Tag filtering auto-enabled: request carries tags but "
                        "enable_tag_filtering is not set. Set enable_tag_filtering=False "
                        "in router_settings to disable. This message is logged once.",
                    )
                    llm_router_instance._tag_filtering_auto_logged = True  # type: ignore[attr-defined]
            verbose_logger.debug(
                "get_deployments_for_tag routing: router_keys: %s", request_tags
            )
            # example this can be router_keys=["free", "custom"]
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

                if is_valid_deployment_tag(deployment_tags, request_tags, match_any):
                    new_healthy_deployments.append(deployment)

                if "default" in deployment_tags:
                    default_deployments.append(deployment)

            if len(new_healthy_deployments) == 0 and len(default_deployments) == 0:
                if llm_router_instance.enable_tag_filtering is True:
                    raise ValueError(
                        f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model={model} and tags={request_tags}"
                    )
                # Without explicit enable_tag_filtering, fall through
                # gracefully to avoid breaking users who have tags on
                # teams/keys for non-routing purposes (e.g. budget tracking).
                return healthy_deployments

            return (
                new_healthy_deployments
                if len(new_healthy_deployments) > 0
                else default_deployments
            )

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
    request_kwargs: Optional[Dict[Any, Any]] = None,
    metadata_variable_name: Literal["metadata", "litellm_metadata"] = "metadata",
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
    if metadata_variable_name in request_kwargs:
        metadata = request_kwargs[metadata_variable_name] or {}
        tags = metadata.get("tags", [])
        return tags if tags is not None else []
    elif "litellm_params" in request_kwargs:
        litellm_params = request_kwargs["litellm_params"] or {}
        _metadata = litellm_params.get(metadata_variable_name, {}) or {}
        tags = _metadata.get("tags", [])
        return tags if tags is not None else []
    return []
