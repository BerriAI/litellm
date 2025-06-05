# What is this?
## Common checks for /v1/models and `/model/info`
from typing import Dict, List, Optional, Set

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
from litellm.router import Router
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
        # provider_models = copy.deepcopy(litellm.models_by_provider[provider])
        for idx, _model in enumerate(provider_models):
            if provider not in _model:
                provider_models[idx] = f"{provider}/{_model}"
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
            ):  # remove access group, unless requested - e.g. when creating a key and trying to see list of models
                idx_to_remove.append(idx)
            new_models.extend(model_access_groups[model])

    for idx in sorted(idx_to_remove, reverse=True):
        all_models.pop(idx)

    all_models.extend(new_models)
    return all_models


def get_key_models(
    user_api_key_dict: UserAPIKeyAuth,
    proxy_model_list: List[str],
    model_access_groups: Dict[str, List[str]],
    include_model_access_groups: Optional[bool] = False,
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
    - If include_model_access_groups is True, it includes the 'keys' of the model_access_groups in the response - {"beta-models": ["gpt-4", "claude-v1"]} -> returns 'beta-models'
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
    all_models = []
    if len(team_models) > 0:
        all_models = team_models
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = team_models
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = proxy_model_list

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=all_models,
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
        provider, model = wildcard_model.split("/", 1)
    except ValueError:  # safely fail
        return []
    # get all known provider models
    wildcard_models = get_provider_models(
        provider=provider, litellm_params=litellm_params
    )
    if wildcard_models is None:
        return []
    if model == "*":
        return wildcard_models or []
    else:
        model_prefix = model.replace("*", "")
        filtered_wildcard_models = [
            wc_model
            for wc_model in wildcard_models
            if wc_model.split("/")[1].startswith(model_prefix)
        ]

        return filtered_wildcard_models


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
