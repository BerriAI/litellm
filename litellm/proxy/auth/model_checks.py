# What is this?
## Common checks for /v1/models and `/model/info`
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _is_wildcard_pattern,
    _model_matches_any_wildcard_pattern_in_list,
    is_model_allowed_by_pattern,
)
from litellm.proxy.utils import PrismaClient
from litellm.router import Router
from litellm.router_utils.fallback_event_handlers import get_fallback_model_group
from litellm.types.router import LiteLLM_Params
from litellm.utils import get_valid_models


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


def get_provider_models(
    provider: str, litellm_params: Optional[LiteLLM_Params] = None
) -> Optional[List[str]]:
    """
    Returns the list of known models by provider
    """
    if provider == "*":
        return get_valid_models(litellm_params=litellm_params)

    if provider in litellm.models_by_provider:
        provider_models = get_valid_models(
            custom_llm_provider=provider, litellm_params=litellm_params
        )
        return provider_models
    return None


def _get_models_from_access_groups(
    model_access_groups: Dict[str, List[str]],
    all_models: List[str],
    include_model_access_groups: Optional[bool] = False,
) -> List[str]:
    idx_to_remove = []
    new_models = []
    for idx, model in enumerate(all_models):
        if model in model_access_groups:
            if (
                not include_model_access_groups
            ):  # remove access group, unless requested - e.g. when creating a key
                idx_to_remove.append(idx)
            new_models.extend(model_access_groups[model])

    for idx in sorted(idx_to_remove, reverse=True):
        all_models.pop(idx)

    all_models.extend(new_models)
    return all_models


async def get_mcp_server_ids(
    user_api_key_dict: UserAPIKeyAuth,
) -> List[str]:
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
        result = await prisma_client.db.litellm_objectpermissiontable.find_unique(
            where={"object_permission_id": user_api_key_dict.object_permission_id},
        )
        if result and result.mcp_servers:
            return result.mcp_servers
        return []
    except Exception:
        return []


def get_key_models(
    user_api_key_dict: UserAPIKeyAuth,
    proxy_model_list: List[str],
    model_access_groups: Dict[str, List[str]],
    include_model_access_groups: Optional[bool] = False,
    only_model_access_groups: Optional[bool] = False,
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
    - If include_model_access_groups is True, it includes the 'keys' of the model_access_groups
      in the response - {"beta-models": ["gpt-4", "claude-v1"]} -> returns 'beta-models'
    """
    all_models: List[str] = []
    if len(user_api_key_dict.models) > 0:
        all_models = list(
            user_api_key_dict.models
        )  # copy to avoid mutating cached objects
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = list(
                user_api_key_dict.team_models
            )  # copy to avoid mutating cached objects
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = list(proxy_model_list)  # copy to avoid mutating caller's list
            if include_model_access_groups:
                all_models.extend(model_access_groups.keys())

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=all_models,
        include_model_access_groups=include_model_access_groups,
    )

    # deduplicate while preserving order
    all_models = list(dict.fromkeys(all_models))

    verbose_proxy_logger.debug("ALL KEY MODELS - {}".format(len(all_models)))
    return all_models


def get_team_models(
    team_models: List[str],
    proxy_model_list: List[str],
    model_access_groups: Dict[str, List[str]],
    include_model_access_groups: Optional[bool] = False,
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
    """
    all_models_set: Set[str] = set()
    if len(team_models) > 0:
        all_models_set.update(team_models)
        if SpecialModelNames.all_team_models.value in all_models_set:
            all_models_set.update(team_models)
        if SpecialModelNames.all_proxy_models.value in all_models_set:
            all_models_set.update(proxy_model_list)
            if include_model_access_groups:
                all_models_set.update(model_access_groups.keys())

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=list(all_models_set),
        include_model_access_groups=include_model_access_groups,
    )

    # deduplicate while preserving order
    all_models = list(dict.fromkeys(all_models))

    verbose_proxy_logger.debug("ALL TEAM MODELS - {}".format(len(all_models)))
    return all_models


def get_complete_model_list(
    key_models: List[str],
    team_models: List[str],
    proxy_model_list: List[str],
    user_model: Optional[str],
    infer_model_from_keys: Optional[bool],
    return_wildcard_routes: Optional[bool] = False,
    llm_router: Optional[Router] = None,
    model_access_groups: Dict[str, List[str]] = {},
    include_model_access_groups: Optional[bool] = False,
    only_model_access_groups: Optional[bool] = False,
) -> List[str]:
    """Logic for returning complete model list for a given key + team pair"""

    """
    - If key list is empty -> defer to team list
    - If team list is empty -> defer to proxy model list

    If list contains wildcard -> return known provider models
    """

    unique_models = []

    def append_unique(models):
        for model in models:
            if model not in unique_models:
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
            valid_models = get_valid_models()
            append_unique(valid_models)

    if only_model_access_groups:
        model_access_groups_to_return: List[str] = []
        for model in unique_models:
            if model in model_access_groups:
                model_access_groups_to_return.append(model)
        return model_access_groups_to_return

    all_wildcard_models = _get_wildcard_models(
        unique_models=unique_models,
        return_wildcard_routes=return_wildcard_routes,
        llm_router=llm_router,
    )

    complete_model_list = unique_models + all_wildcard_models

    return complete_model_list


def get_known_models_from_wildcard(
    wildcard_model: str, litellm_params: Optional[LiteLLM_Params] = None
) -> List[str]:
    try:
        wildcard_provider_prefix, wildcard_suffix = wildcard_model.split("/", 1)
    except ValueError:  # safely fail
        return []

    # Use provider from litellm_params when available, otherwise from wildcard prefix
    # (e.g., "openai" from "openai/*" - needed for BYOK where wildcard isn't in router)
    if litellm_params is not None:
        try:
            provider = litellm_params.model.split("/", 1)[0]
        except ValueError:
            provider = wildcard_provider_prefix
    else:
        provider = wildcard_provider_prefix

    # get all known provider models

    wildcard_models = get_provider_models(
        provider=provider, litellm_params=litellm_params
    )

    if wildcard_models is None:
        return []
    if wildcard_suffix != "*":
        ## CHECK IF PARTIAL FILTER e.g. `gemini-*`
        model_prefix = wildcard_suffix.replace("*", "")

        is_partial_filter = any(
            wc_model.startswith(model_prefix) for wc_model in wildcard_models
        )
        if is_partial_filter:
            filtered_wildcard_models = [
                wc_model
                for wc_model in wildcard_models
                if wc_model.startswith(model_prefix)
            ]
            wildcard_models = filtered_wildcard_models
        else:
            # add model prefix to wildcard models
            wildcard_models = [f"{model_prefix}{model}" for model in wildcard_models]

    suffix_appended_wildcard_models = []
    for model in wildcard_models:
        if not model.startswith(wildcard_provider_prefix):
            model = f"{wildcard_provider_prefix}/{model}"
        suffix_appended_wildcard_models.append(model)
    return suffix_appended_wildcard_models or []


def _get_wildcard_models(
    unique_models: List[str],
    return_wildcard_routes: Optional[bool] = False,
    llm_router: Optional[Router] = None,
) -> List[str]:
    models_to_remove = set()
    all_wildcard_models = []
    for model in unique_models:
        if _check_wildcard_routing(model=model):
            if (
                return_wildcard_routes
            ):  # will add the wildcard route to the list eg: anthropic/*.
                all_wildcard_models.append(model)

            ## get litellm params from model
            if llm_router is not None:
                model_list = llm_router.get_model_list(model_name=model)
                if model_list:
                    for router_model in model_list:
                        wildcard_models = get_known_models_from_wildcard(
                            wildcard_model=model,
                            litellm_params=LiteLLM_Params(
                                **router_model["litellm_params"]  # type: ignore
                            ),
                        )
                        all_wildcard_models.extend(wildcard_models)
                else:
                    # Router has no deployment for this wildcard (e.g., BYOK team models)
                    # Fall back to expanding from known provider models
                    wildcard_models = get_known_models_from_wildcard(
                        wildcard_model=model, litellm_params=None
                    )
                    if wildcard_models:
                        models_to_remove.add(model)
                        all_wildcard_models.extend(wildcard_models)
            else:
                # get all known provider models
                wildcard_models = get_known_models_from_wildcard(
                    wildcard_model=model, litellm_params=None
                )

                if wildcard_models:
                    models_to_remove.add(model)
                    all_wildcard_models.extend(wildcard_models)

    for model in models_to_remove:
        unique_models.remove(model)

    return all_wildcard_models


def get_all_fallbacks(
    model: str,
    llm_router: Optional[Router] = None,
    fallback_type: str = "general",
) -> List[str]:
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
        verbose_proxy_logger.warning(f"Unknown fallback_type: {fallback_type}")
        return []

    if not fallbacks_config:
        return []

    try:
        # Use existing function to get fallback model group
        fallback_model_group, _ = get_fallback_model_group(
            fallbacks=fallbacks_config, model_group=model
        )

        if fallback_model_group is None:
            return []

        return fallback_model_group
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting fallbacks for model {model}: {e}")
        return []


# --- GET /key/{key_id}/models (admin UI): resolve lists and sectioned payload ---

KEY_RESOLVED_MODELS_DISPLAY_LIMIT = 500

_KEY_SECTION_ALL_PROXY = "all_proxy_models"
_KEY_SECTION_ALL_TEAM = "all_team_models"
_KEY_SECTION_ACCESS_GROUP = "access_group"
_KEY_SECTION_UNGROUPED = "ungrouped"

_KEY_TITLE_ALL_PROXY = "All proxy models"
_KEY_TITLE_ALL_TEAM = "All team models"
_KEY_TITLE_UNGROUPED = "Other models"


class KeyResolvedModelDisplaySection(TypedDict):
    title: str
    section_kind: str
    models: List[str]


def _filter_key_models_by_search(models: List[str], search: Optional[str]) -> List[str]:
    if not search:
        return list(models)
    needle = search.strip().lower()
    if not needle:
        return list(models)
    return [m for m in models if needle in m.lower()]


async def resolve_key_models_for_display(
    *,
    key_models: List[str],
    team_id: Optional[str],
    prisma_client: PrismaClient,
    llm_router: Optional[Router],
) -> Tuple[List[str], str, bool]:
    """
    Returns (resolved_model_names, source, all_team_models_without_team).
    """
    all_models: List[str] = []
    if llm_router is not None:
        all_models = list(llm_router.get_model_names())

    source: str = SpecialModelNames.no_default_models.value
    resolved: List[str] = list(key_models)
    all_team_models_without_team = False

    if SpecialModelNames.all_team_models.value in key_models:
        if team_id is not None:
            source = SpecialModelNames.all_team_models.value
            team_row = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": team_id},
            )
            if team_row is not None and team_row.models is not None:
                resolved = list(team_row.models)
        else:
            source = SpecialModelNames.all_team_models.value
            all_team_models_without_team = True
            resolved = list(all_models)

    if (
        SpecialModelNames.all_proxy_models.value in key_models
        or SpecialModelNames.all_proxy_models.value in resolved
    ):
        source = SpecialModelNames.all_proxy_models.value
        resolved = list(all_models)

    return resolved, source, all_team_models_without_team


def _concrete_models_allowed_by_resolved(
    resolved: List[str], router_model_names: List[str]
) -> List[str]:
    """Concrete router model names the key may call, given resolved patterns (incl. wildcards)."""
    resolved_list = list(resolved)
    resolved_set = set(resolved)
    out: List[str] = []
    seen: set[str] = set()
    for m in router_model_names:
        if m in seen:
            continue
        if m in resolved_set:
            out.append(m)
            seen.add(m)
        elif _model_matches_any_wildcard_pattern_in_list(m, resolved_list):
            out.append(m)
            seen.add(m)
    return out


def _expand_group_models_for_key_display(
    group_models: List[str],
    concrete_pool: List[str],
) -> List[str]:
    """
    Expand wildcard entries in a group's model list to concrete names from the pool.
    Non-wildcard entries are included when present in the pool.
    """
    pool_set = set(concrete_pool)
    out: List[str] = []
    seen: set[str] = set()
    for g in group_models:
        if _is_wildcard_pattern(g):
            for m in concrete_pool:
                if m in seen:
                    continue
                if is_model_allowed_by_pattern(m, g):
                    out.append(m)
                    seen.add(m)
        else:
            if g in seen:
                continue
            if g in pool_set:
                out.append(g)
                seen.add(g)
    return out


def _models_in_any_access_group_for_key_display(
    model_access_groups: Dict[str, List[str]], candidate_models: List[str]
) -> set:
    """Set of candidate_models that appear in at least one access group list (concrete names)."""
    cset = set(candidate_models)
    in_group: set = set()
    for models_in_group in model_access_groups.values():
        for m in models_in_group:
            if m in cset:
                in_group.add(m)
    return in_group


def build_key_resolved_model_display_sections(
    *,
    display_models: List[str],
    source: str,
    model_access_groups: Dict[str, List[str]],
    compact: bool,
) -> List[KeyResolvedModelDisplaySection]:
    """
    Build ordered sections for the admin UI.

    When source is all-proxy or all-team, a scope section lists all display_models first,
    then every intersecting **model_access_groups** section is still included (models may
    repeat across the sentinel section and each group). Ungrouped is omitted for sentinels
    only to avoid repeating the full flat list as "Other models".
    """
    sections: List[KeyResolvedModelDisplaySection] = []
    empty_models: List[str] = [] if compact else []

    display_set = set(display_models)

    if source == SpecialModelNames.all_proxy_models.value:
        sections.append(
            KeyResolvedModelDisplaySection(
                title=_KEY_TITLE_ALL_PROXY,
                section_kind=_KEY_SECTION_ALL_PROXY,
                models=empty_models if compact else list(display_models),
            )
        )
    elif source == SpecialModelNames.all_team_models.value:
        sections.append(
            KeyResolvedModelDisplaySection(
                title=_KEY_TITLE_ALL_TEAM,
                section_kind=_KEY_SECTION_ALL_TEAM,
                models=empty_models if compact else list(display_models),
            )
        )

    for group_name, group_models in model_access_groups.items():
        intersected = [m for m in group_models if m in display_set]
        if not intersected:
            continue
        sections.append(
            KeyResolvedModelDisplaySection(
                title=group_name,
                section_kind=_KEY_SECTION_ACCESS_GROUP,
                models=empty_models if compact else intersected,
            )
        )

    skip_ungrouped = source in (
        SpecialModelNames.all_proxy_models.value,
        SpecialModelNames.all_team_models.value,
    )
    if not skip_ungrouped:
        in_any = _models_in_any_access_group_for_key_display(
            model_access_groups, display_models
        )
        ungrouped = [m for m in display_models if m not in in_any]
        if ungrouped:
            sections.append(
                KeyResolvedModelDisplaySection(
                    title=_KEY_TITLE_UNGROUPED,
                    section_kind=_KEY_SECTION_UNGROUPED,
                    models=empty_models if compact else ungrouped,
                )
            )

    return sections


def prepare_key_models_response_payload(
    *,
    resolved: List[str],
    source: str,
    all_team_models_without_team: bool,
    model_access_groups: Dict[str, List[str]],
    search: Optional[str],
    compact: bool,
    all_router_model_names: List[str],
) -> Dict[str, Any]:
    """
    Apply search, truncation, and build the JSON-serializable response dict for GET /key/{id}/models.

    Access-group sections list concrete model names: wildcards in router group metadata
    are expanded against models allowed for this key (resolved ∩ router names).
    """
    base_concrete = _concrete_models_allowed_by_resolved(resolved, all_router_model_names)

    expanded_groups: Dict[str, List[str]] = {}
    for group_name, group_models in model_access_groups.items():
        expanded = _expand_group_models_for_key_display(group_models, base_concrete)
        if expanded:
            expanded_groups[group_name] = expanded

    filtered_concrete = _filter_key_models_by_search(base_concrete, search)
    matched_count = len(filtered_concrete)
    models_truncated = matched_count > KEY_RESOLVED_MODELS_DISPLAY_LIMIT
    display_models = (
        filtered_concrete[:KEY_RESOLVED_MODELS_DISPLAY_LIMIT]
        if models_truncated
        else filtered_concrete
    )

    display_set = set(display_models)
    intersected_groups: Dict[str, List[str]] = {}
    for group_name, models in expanded_groups.items():
        in_slice = [m for m in models if m in display_set]
        if in_slice:
            intersected_groups[group_name] = in_slice

    model_display_sections = build_key_resolved_model_display_sections(
        display_models=display_models,
        source=source,
        model_access_groups=intersected_groups,
        compact=compact,
    )

    return {
        "model_display_sections": model_display_sections,
        "source": source,
        "resolved_total_count": len(resolved),
        "matched_count": matched_count,
        "models_truncated": models_truncated,
        "all_team_models_without_team": all_team_models_without_team,
    }
