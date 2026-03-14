"""
Use this to route requests between Teams

- If tags in request is a subset of tags in deployment, return deployment
- if deployments are set with default tags, return all default deployment
- If no default_deployments are set, return all deployments
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.router import RouterErrors

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _is_valid_deployment_tag_regex(
    tag_regexes: List[str],
    header_strings: List[str],
) -> Optional[str]:
    """
    Test compiled regex patterns against "Header-Name: value" strings.

    Returns the first matching pattern string, or None if nothing matches.
    Compiles each pattern once (re's LRU cache) and logs invalid patterns once
    per pattern, not once per header string.
    """
    for pattern in tag_regexes:
        try:
            compiled = re.compile(pattern)
        except re.error:
            verbose_logger.warning(
                "tag_regex: invalid pattern %r — skipping", pattern
            )
            continue
        for header_str in header_strings:
            if compiled.search(header_str):
                return pattern
    return None


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


def _match_deployment(
    deployment: Any,
    request_tags: Optional[List[str]],
    header_strings: List[str],
    match_any: bool,
) -> Optional[Dict[str, str]]:
    """
    Determine whether *deployment* matches the current request.

    Returns {"matched_via": ..., "matched_value": ...} if the deployment
    should be included, or None if it should be excluded.

    Priority:
      1. Exact tag match (respects match_any semantics).
      2. Regex match — skipped when match_any=False and the tag check already
         ran and failed, so the regex cannot override strict-tag policy.
    """
    litellm_params = deployment.get("litellm_params", {})
    deployment_tags: Optional[List[str]] = litellm_params.get("tags")
    deployment_tag_regex: Optional[List[str]] = litellm_params.get("tag_regex")

    # 1. Exact tag match (existing behaviour).
    if deployment_tags and request_tags:
        if is_valid_deployment_tag(deployment_tags, request_tags, match_any):
            matched_value = next(
                (t for t in deployment_tags if t in set(request_tags)),
                deployment_tags[0],
            )
            return {"matched_via": "tags", "matched_value": matched_value}

    # 2. Regex match against request headers.
    # When match_any=False and the deployment has both plain tags and tag_regex,
    # the strict tag check has already failed (step 1 returned None).  Allow
    # the regex to fire only when the deployment has NO plain tags, so we never
    # use regex as a backdoor around the operator's strict-tag policy.
    strict_tag_check_failed = (
        not match_any
        and bool(deployment_tags)
        and bool(request_tags)
    )
    if deployment_tag_regex and header_strings and not strict_tag_check_failed:
        regex_match = _is_valid_deployment_tag_regex(deployment_tag_regex, header_strings)
        if regex_match is not None:
            return {"matched_via": "tag_regex", "matched_value": regex_match}

    return None


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

    verbose_logger.debug(
        "request metadata: %s", request_kwargs.get(metadata_variable_name)
    )
    if metadata_variable_name in request_kwargs:
        metadata = request_kwargs[metadata_variable_name]
        request_tags = metadata.get("tags")
        match_any = llm_router_instance.tag_filtering_match_any

        # Build header strings for regex matching from what the proxy already stores.
        # Currently we match against User-Agent; format matches "^User-Agent: claude-code/..."
        user_agent = metadata.get("user_agent", "")
        header_strings: List[str] = (
            [f"User-Agent: {user_agent}"] if user_agent else []
        )

        new_healthy_deployments: List[Any] = []
        default_deployments: List[Any] = []

        # Only activate header-based regex filtering when at least one deployment in
        # the candidate set has tag_regex configured.  This preserves existing
        # behaviour for operators who use plain tags: a request that carries a
        # User-Agent (all proxy requests do) but targets deployments with no
        # tag_regex will continue to use the original tag-only code path.
        has_regex_deployments = any(
            d.get("litellm_params", {}).get("tag_regex")
            for d in healthy_deployments
        )
        has_tag_filter = bool(request_tags) or (
            bool(header_strings) and has_regex_deployments
        )
        if has_tag_filter:
            verbose_logger.debug(
                "get_deployments_for_tag routing: request_tags=%s user_agent=%s",
                request_tags,
                user_agent,
            )
            for deployment in healthy_deployments:
                deployment_tags = deployment.get("litellm_params", {}).get("tags")

                match_result = _match_deployment(
                    deployment=deployment,
                    request_tags=request_tags,
                    header_strings=header_strings,
                    match_any=match_any,
                )

                if match_result is not None:
                    verbose_logger.debug(
                        "tag routing match: deployment=%s matched_via=%s matched_value=%s",
                        deployment.get("model_name"),
                        match_result["matched_via"],
                        match_result["matched_value"],
                    )
                    # Record provenance in metadata so it flows to SpendLogs.
                    # Written only for the first match — load balancer selects one
                    # deployment from new_healthy_deployments, so overwriting on
                    # subsequent matches would produce misleading observability data.
                    if "tag_routing" not in metadata:
                        metadata["tag_routing"] = {
                            "matched_deployment": deployment.get("model_name"),
                            "matched_via": match_result["matched_via"],
                            "matched_value": match_result["matched_value"],
                            "request_tags": request_tags or [],
                            "user_agent": user_agent,
                        }
                    new_healthy_deployments.append(deployment)

                if deployment_tags and "default" in deployment_tags:
                    default_deployments.append(deployment)

            if len(new_healthy_deployments) == 0 and len(default_deployments) == 0:
                raise ValueError(
                    f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model={model} and tags={request_tags}"
                )

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
