# What is this?
## Common checks for /v1/models and `/model/info`
import copy
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_llm_provider_logic import (
    model_matches_wildcard,
    split_route_type_prefix,
    strip_provider_family_prefix,
)
from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.router import Router
from litellm.router_utils.fallback_event_handlers import get_fallback_model_group
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
from litellm.types.router import CredentialLiteLLMParams, LiteLLM_Params
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_valid_models

_CREDENTIAL_LITELLM_PARAM_FIELDS = set(CredentialLiteLLMParams.model_fields)


_CREDENTIAL_LITELLM_PARAM_FIELDS = set(CredentialLiteLLMParams.model_fields)


def _check_wildcard_routing(model: str) -> bool:
    """
    Returns True if a model is a provider wildcard.

    eg:
    - anthropic/*
    - openai/*
    - *
    """
    if "*" in model:
        return True
    return False


def get_provider_models(provider: str, litellm_params: LiteLLM_Params | None = None) -> list[str] | None:
    """
    Returns the list of known models by provider
    """
    if provider == "*":
        return get_valid_models(litellm_params=litellm_params)

    if provider in litellm.models_by_provider:
        provider_models: Final = get_valid_models(custom_llm_provider=provider, litellm_params=litellm_params)
        return provider_models

    try:
        llm_provider: Final = LlmProviders(provider)
    except ValueError:
        return None
    if ProviderConfigManager.get_provider_model_info(model=None, provider=llm_provider) is None:
        return None
    return get_valid_models(custom_llm_provider=provider, litellm_params=litellm_params)


def _get_models_from_access_groups(
    model_access_groups: dict[str, list[str]],
    all_models: list[str],
    include_model_access_groups: bool | None = False,
    proxy_model_list: Sequence[str] | None = None,
) -> list[str]:
    # a grant naming both a deployed model and an access group means both at runtime
    # (_check_model_access_helper unions them), so listings must keep the literal too
    deployed_model_names: Final = frozenset(proxy_model_list or ())
    kept_models: Final = [
        model
        for model in all_models
        if model not in model_access_groups or include_model_access_groups or model in deployed_model_names
    ]
    member_models: Final = [
        member for model in all_models if model in model_access_groups for member in model_access_groups[model]
    ]
    return kept_models + member_models


async def get_mcp_server_ids(
    user_api_key_dict: UserAPIKeyAuth,
) -> list[str]:
    """
    Returns the list of MCP server ids for a given key by querying the object_permission table
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return []

    if user_api_key_dict.object_permission_id is None:
        return []

    # Make a direct SQL query to get just the mcp_servers
    try:
        result: Final = await ObjectPermissionRepository(prisma_client).table.find_unique(
            where={"object_permission_id": user_api_key_dict.object_permission_id},
        )
        if result and result.mcp_servers:
            return result.mcp_servers
        return []
    except Exception:
        return []


def get_key_models(
    user_api_key_dict: UserAPIKeyAuth,
    proxy_model_list: list[str],
    model_access_groups: dict[str, list[str]],
    include_model_access_groups: bool | None = False,
    only_model_access_groups: bool | None = False,
) -> list[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
    - If include_model_access_groups is True, it includes the 'keys' of the model_access_groups
      in the response - {"beta-models": ["gpt-4", "claude-v1"]} -> returns 'beta-models'
    """
    all_models: list[str] = []
    if len(user_api_key_dict.models) > 0:
        all_models = list(user_api_key_dict.models)  # copy to avoid mutating cached objects
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = list(user_api_key_dict.team_models)
            if SpecialModelNames.all_team_models.value in all_models:
                all_models = [model for model in all_models if model != SpecialModelNames.all_team_models.value]
                all_models.extend(proxy_model_list)
                if include_model_access_groups:
                    all_models.extend(model_access_groups.keys())
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = list(proxy_model_list)  # copy to avoid mutating caller's list
            if include_model_access_groups:
                all_models.extend(model_access_groups.keys())

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=all_models,
        include_model_access_groups=include_model_access_groups,
        proxy_model_list=proxy_model_list,
    )

    # deduplicate while preserving order
    all_models = list(dict.fromkeys(all_models))

    verbose_proxy_logger.debug("ALL KEY MODELS - %s", len(all_models))
    return all_models


def get_team_models(
    team_models: list[str],
    proxy_model_list: list[str],
    model_access_groups: dict[str, list[str]],
    include_model_access_groups: bool | None = False,
) -> list[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
    """
    all_models_set: Final[set[str]] = set()
    if len(team_models) > 0:
        all_models_set.update(team_models)
        if SpecialModelNames.all_team_models.value in all_models_set:
            all_models_set.update(team_models)
            # GH#30619: expand all-team-models sentinel
            # to the actual proxy model list
            all_models_set.discard(SpecialModelNames.all_team_models.value)
            all_models_set.update(proxy_model_list)
            if include_model_access_groups:
                all_models_set.update(model_access_groups.keys())
        if SpecialModelNames.all_proxy_models.value in all_models_set:
            all_models_set.update(proxy_model_list)
            if include_model_access_groups:
                all_models_set.update(model_access_groups.keys())

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=list(all_models_set),
        include_model_access_groups=include_model_access_groups,
        proxy_model_list=proxy_model_list,
    )

    # deduplicate while preserving order
    all_models = list(dict.fromkeys(all_models))

    verbose_proxy_logger.debug("ALL TEAM MODELS - %s", len(all_models))
    return all_models


def get_complete_model_list(
    key_models: Sequence[str],
    team_models: Sequence[str],
    proxy_model_list: list[str],
    user_model: str | None,
    infer_model_from_keys: bool | None,
    return_wildcard_routes: bool | None = False,
    llm_router: Router | None = None,
    model_access_groups: dict[str, list[str]] = {},
    include_model_access_groups: bool | None = False,
    only_model_access_groups: bool | None = False,
    team_id: str | None = None,
) -> list[str]:
    """Logic for returning complete model list for a given key + team pair"""

    """
    - If key list is empty -> defer to team list
    - If team list is empty -> defer to proxy model list

    If list contains wildcard -> return known provider models
    """

    unique_models: Final = []

    def append_unique(models):
        for model in models:
            if model not in unique_models and model != SpecialModelNames.no_default_models.value:
                unique_models.append(model)

    if key_models:
        append_unique(key_models)
    elif team_models:
        append_unique(team_models)
    else:
        append_unique(proxy_model_list)
        if include_model_access_groups:
            append_unique(list(model_access_groups.keys()))  # TODO: keys order

        if user_model:
            append_unique([user_model])

        if infer_model_from_keys:
            valid_models: Final = get_valid_models()
            append_unique(valid_models)

    if only_model_access_groups:
        model_access_groups_to_return: Final[list[str]] = []
        for model in unique_models:
            if model in model_access_groups:
                model_access_groups_to_return.append(model)
        return model_access_groups_to_return

    all_wildcard_models: Final = _get_wildcard_models(
        unique_models=unique_models,
        return_wildcard_routes=return_wildcard_routes,
        llm_router=llm_router,
        team_id=team_id,
    )

    return list(dict.fromkeys(chain(unique_models, all_wildcard_models)))


def _hydrate_litellm_credential_name(
    litellm_params: LiteLLM_Params | None,
) -> LiteLLM_Params | None:
    if litellm_params is None or litellm_params.litellm_credential_name is None:
        return litellm_params

    credential_values: Final = CredentialAccessor.get_credential_values(litellm_params.litellm_credential_name)
    if not credential_values:
        return litellm_params

    litellm_params = litellm_params.model_copy()
    for key, value in credential_values.items():
        if key in _CREDENTIAL_LITELLM_PARAM_FIELDS and getattr(litellm_params, key, None) is None:
            setattr(litellm_params, key, value)
    litellm_params.litellm_credential_name = None
    return litellm_params


@dataclass(frozen=True, slots=True)
class _WildcardListing:
    listed_id: str
    routed_model: str


def _single_star_listing(
    route: str, provider: str, deployment_path: str, model_pattern: str, model_id: str
) -> _WildcardListing | None:
    model_prefix, _, model_suffix = model_pattern.partition("*")
    capture_end: Final = len(model_id) - len(model_suffix)
    if capture_end < len(model_prefix) or not (model_id.startswith(model_prefix) and model_id.endswith(model_suffix)):
        return None
    capture: Final = model_id[len(model_prefix) : capture_end]
    return _WildcardListing(
        listed_id=route.replace("*", capture, 1),
        routed_model=f"{provider}/{deployment_path.replace('*', capture, 1)}",
    )


def _same_pattern_listing(
    route_namespace: str, provider: str, route_type_prefix: str, model_pattern: str, model_id: str
) -> _WildcardListing | None:
    if not model_matches_wildcard(model_id, model_pattern):
        return None
    return _WildcardListing(
        listed_id=f"{route_namespace}{model_id}", routed_model=f"{provider}/{route_type_prefix}{model_id}"
    )


def _wildcard_listings(wildcard_model: str, litellm_params: LiteLLM_Params | None) -> tuple[_WildcardListing, ...]:
    deployment_model: Final = litellm_params.model if litellm_params is not None else wildcard_model
    provider, provider_sep, deployment_path = deployment_model.partition("/")
    if not provider_sep:
        return ()

    route_type_prefix, model_pattern = split_route_type_prefix(deployment_path, provider)
    if wildcard_model.count("*") == 1 and "*" not in model_pattern:
        return (_WildcardListing(listed_id=wildcard_model.replace("*", model_pattern), routed_model=deployment_model),)

    route_namespace: Final = wildcard_model.removesuffix(model_pattern)
    is_single_star: Final = wildcard_model.count("*") == 1 and model_pattern.count("*") == 1
    is_same_pattern: Final = (
        wildcard_model.endswith(model_pattern) and route_namespace.endswith("/") and "*" not in route_namespace
    )
    if not (is_single_star or is_same_pattern):
        return ()

    provider_models: Final = get_provider_models(
        provider=provider, litellm_params=_hydrate_litellm_credential_name(litellm_params)
    )
    if provider_models is None:
        return ()

    model_ids: Final = (strip_provider_family_prefix(model, provider) for model in provider_models)
    listings: Final = (
        _single_star_listing(wildcard_model, provider, deployment_path, model_pattern, model_id)
        if is_single_star
        else _same_pattern_listing(route_namespace, provider, route_type_prefix, model_pattern, model_id)
        for model_id in model_ids
    )
    return tuple(dict.fromkeys(filter(None, listings)))


def _listings_served_by_route(
    wildcard_model: str, litellm_params: LiteLLM_Params | None, pattern_router: PatternMatchRouter | None
) -> tuple[_WildcardListing, ...]:
    listings: Final = _wildcard_listings(wildcard_model, litellm_params)
    if pattern_router is None:
        return listings
    return tuple(
        listing
        for listing in listings
        if not pattern_router.routes_elsewhere(listing.listed_id, wildcard_model, listing.routed_model)
    )


@dataclass(frozen=True, slots=True)
class _ModelInfoRow:
    listing: _WildcardListing
    shadowed: bool


def _model_info_rows(
    deployment: dict[str, object], pattern_router: PatternMatchRouter | None, exact_model_names: frozenset[str]
) -> tuple[_ModelInfoRow, ...]:
    model_name: Final = str(deployment.get("model_name") or "")
    raw_params: Final = deployment.get("litellm_params")
    if not _check_wildcard_routing(model_name):
        return ()
    try:
        litellm_params: Final = (
            LiteLLM_Params.model_validate(raw_params) if isinstance(raw_params, dict) and raw_params else None
        )
    except Exception:
        return ()
    return tuple(
        _ModelInfoRow(
            listing=listing,
            shadowed=pattern_router is not None
            and pattern_router.shadowing_regex(listing.listed_id, model_name) is not None,
        )
        for listing in _listings_served_by_route(model_name, litellm_params, pattern_router)
        if listing.listed_id not in exact_model_names
    )


def get_known_models_from_wildcard(wildcard_model: str, litellm_params: LiteLLM_Params | None = None) -> list[str]:
    return [listing.listed_id for listing in _wildcard_listings(wildcard_model, litellm_params)]


def expand_wildcard_deployments_for_model_info(
    deployments: list[dict[str, object]],
    pattern_router: PatternMatchRouter | None = None,
) -> list[dict[str, object]]:
    """Expand wildcard deployments into one row per known provider model.

    PR #30025 changed /model/info to read from llm_router.model_list (correct,
    so team-scoped rows are included). This function restores wildcard expansion
    on top of that: a wildcard deployment like model_name="*" / litellm_params.model="openai/*"
    becomes one entry per known openai model, matching /v1/models behaviour.
    """
    model_names: Final = tuple(str(deployment.get("model_name") or "") for deployment in deployments)
    exact_model_names: Final = frozenset(name for name in model_names if not _check_wildcard_routing(name))
    rows_by_deployment: Final = tuple(
        _model_info_rows(deployment, pattern_router, exact_model_names) for deployment in deployments
    )
    served_ids: Final = frozenset(
        candidate.listing.listed_id for candidate in chain.from_iterable(rows_by_deployment) if not candidate.shadowed
    )
    expanded: Final[list[dict[str, object]]] = []
    for deployment, candidates in zip(deployments, rows_by_deployment):
        listings = tuple(
            candidate.listing
            for candidate in candidates
            if not (candidate.shadowed and candidate.listing.listed_id in served_ids)
        )
        if not listings:
            expanded.append(deployment)
            continue

        for listing in listings:
            row = copy.deepcopy(deployment)
            row["model_name"] = listing.listed_id
            params = row.get("litellm_params")
            if isinstance(params, dict):
                params["model"] = listing.routed_model
            expanded.append(row)

    return expanded


def _ids_listed_by_route(route: str, llm_router: Router | None, team_id: str | None) -> tuple[str, ...]:
    model_list: Final = llm_router.get_model_list(model_name=route, team_id=team_id) if llm_router else None
    pattern_router: Final = llm_router.pattern_router if llm_router else None
    listings: Final = (
        chain.from_iterable(
            _listings_served_by_route(route, LiteLLM_Params(**router_model["litellm_params"]), pattern_router)
            for router_model in model_list
        )
        if model_list
        else _listings_served_by_route(route, None, pattern_router)
    )
    return tuple(listing.listed_id for listing in listings)


def _routes_can_list_same_ids(route: str, other_route: str) -> bool:
    prefix: Final = route.partition("*")[0]
    other_prefix: Final = other_route.partition("*")[0]
    return prefix.startswith(other_prefix) or other_prefix.startswith(prefix)


def _ids_other_routes_serve_under(
    wildcards: tuple[str, ...], llm_router: Router | None, team_id: str | None
) -> tuple[str, ...]:
    if llm_router is None or not wildcards:
        return ()
    other_routes: Final = tuple(
        dict.fromkeys(
            route
            for route in llm_router.get_model_names(team_id=team_id)
            if _check_wildcard_routing(route) and route not in wildcards
        )
    )
    overlapping_routes: Final = (
        route for route in other_routes if any(_routes_can_list_same_ids(route, wildcard) for wildcard in wildcards)
    )
    other_ids: Final = chain.from_iterable(
        _ids_listed_by_route(route, llm_router, team_id) for route in overlapping_routes
    )
    return tuple(
        listed_id
        for listed_id in other_ids
        if any(model_matches_wildcard(listed_id, wildcard) for wildcard in wildcards)
    )


def _get_wildcard_models(
    unique_models: list[str],
    return_wildcard_routes: bool | None = False,
    llm_router: Router | None = None,
    team_id: str | None = None,
) -> list[str]:
    wildcard_routes: Final = tuple(model for model in unique_models if _check_wildcard_routing(model=model))
    for model in wildcard_routes:
        unique_models.remove(model)
    listed: Final = chain.from_iterable(
        (*((model,) if return_wildcard_routes else ()), *_ids_listed_by_route(model, llm_router, team_id))
        for model in wildcard_routes
    )
    return list(dict.fromkeys(chain(listed, _ids_other_routes_serve_under(wildcard_routes, llm_router, team_id))))


def get_all_fallbacks(
    model: str,
    llm_router: Router | None = None,
    fallback_type: str = "general",
) -> list[str]:
    """
    Get all fallbacks for a given model from the router's fallback configuration.

    Args:
        model: The model name to get fallbacks for
        llm_router: The LiteLLM router instance
        fallback_type: Type of fallback ("general", "context_window", "content_policy")

    Returns:
        List of fallback model names. Empty list if no fallbacks found.
    """
    if llm_router is None:
        return []

    # Get the appropriate fallback list based on type
    fallbacks_config: list = []
    if fallback_type == "general":
        fallbacks_config = getattr(llm_router, "fallbacks", [])
    elif fallback_type == "context_window":
        fallbacks_config = getattr(llm_router, "context_window_fallbacks", [])
    elif fallback_type == "content_policy":
        fallbacks_config = getattr(llm_router, "content_policy_fallbacks", [])
    else:
        verbose_proxy_logger.warning("Unknown fallback_type: %s", fallback_type)
        return []

    if not fallbacks_config:
        return []

    try:
        # Use existing function to get fallback model group
        fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks_config, model_group=model)

        if fallback_model_group is None:
            return []

        return fallback_model_group
    except Exception as e:
        verbose_proxy_logger.error("Error getting fallbacks for model %s: %s", model, e)
        return []
