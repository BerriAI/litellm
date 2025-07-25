# What is this?
## Common checks for /v1/models and `/model/info`
from typing import Dict, List, Optional, Set

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
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
        all_models = user_api_key_dict.models
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = user_api_key_dict.team_models
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = proxy_model_list

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups, all_models=all_models
    )

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

    all_models = list(all_models_set)

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=list(all_models_set),
        include_model_access_groups=include_model_access_groups,
    )

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

    unique_models: Set[str] = set()
    if key_models:
        unique_models.update(key_models)
    elif team_models:
        unique_models.update(team_models)
    else:
        unique_models.update(proxy_model_list)
        if include_model_access_groups:
            unique_models.update(model_access_groups.keys())

        if user_model:
            unique_models.add(user_model)

        if infer_model_from_keys:
            valid_models = get_valid_models()
            unique_models.update(valid_models)

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

    complete_model_list = list(unique_models) + all_wildcard_models

    return complete_model_list


def get_known_models_from_wildcard(
    wildcard_model: str, litellm_params: Optional[LiteLLM_Params] = None
) -> List[str]:
    try:
        wildcard_provider_prefix, wildcard_suffix = wildcard_model.split("/", 1)
    except ValueError:  # safely fail
        return []

    if litellm_params is None:  # need litellm params to extract litellm model name
        return []

    try:
        provider = litellm_params.model.split("/", 1)[0]
    except ValueError:
        provider = wildcard_provider_prefix

    # get all known provider models
    wildcard_models = get_provider_models(
        provider=provider, litellm_params=litellm_params
    )
    if wildcard_models is None:
        return []
    if wildcard_suffix != "*":
        model_prefix = wildcard_suffix.replace("*", "")
        filtered_wildcard_models = [
            wc_model
            for wc_model in wildcard_models
            if wc_model.startswith(model_prefix)
        ]
        wildcard_models = filtered_wildcard_models

    suffix_appended_wildcard_models = []
    for model in wildcard_models:
        if not model.startswith(wildcard_provider_prefix):
            model = f"{wildcard_provider_prefix}/{model}"
        suffix_appended_wildcard_models.append(model)
    return suffix_appended_wildcard_models or []


def _get_wildcard_models(
    unique_models: Set[str],
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
                if model_list is not None:
                    for router_model in model_list:
                        wildcard_models = get_known_models_from_wildcard(
                            wildcard_model=model,
                            litellm_params=LiteLLM_Params(
                                **router_model["litellm_params"]  # type: ignore
                            ),
                        )
                        all_wildcard_models.extend(wildcard_models)
            else:
                # get all known provider models
                wildcard_models = get_known_models_from_wildcard(wildcard_model=model)

                if wildcard_models is not None:
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
            fallbacks=fallbacks_config, 
            model_group=model
        )
        
        if fallback_model_group is None:
            return []
        
        return fallback_model_group
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting fallbacks for model {model}: {e}")
        return []
