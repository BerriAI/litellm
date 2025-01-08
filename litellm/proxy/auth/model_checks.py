# What is this?
## Common checks for /v1/models and `/model/info`
from typing import Dict, List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
from litellm.utils import get_valid_models


def _check_wildcard_routing(model: str) -> bool:
    """
    Returns True if a model is a provider wildcard.
    """
    if model == "*":
        return True

    if "/" in model:
        llm_provider, potential_wildcard = model.split("/", 1)
        if (
            llm_provider in litellm.provider_list and potential_wildcard == "*"
        ):  # e.g. anthropic/*
            return True

    return False


def get_provider_models(provider: str) -> Optional[List[str]]:
    """
    Returns the list of known models by provider
    """
    if provider == "*":
        return get_valid_models()

    if provider in litellm.models_by_provider:
        provider_models = litellm.models_by_provider[provider]
        for idx, _model in enumerate(provider_models):
            if provider not in _model:
                provider_models[idx] = f"{provider}/{_model}"
        return provider_models
    return None


def _get_models_from_access_groups(
    model_access_groups: Dict[str, List[str]],
    all_models: List[str],
) -> List[str]:
    idx_to_remove = []
    new_models = []
    for idx, model in enumerate(all_models):
        if model in model_access_groups:
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
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
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
    user_api_key_dict: UserAPIKeyAuth,
    proxy_model_list: List[str],
    model_access_groups: Dict[str, List[str]],
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    - If model_access_groups is provided, only return models that are in the access groups
    """
    all_models = []
    if len(user_api_key_dict.team_models) > 0:
        all_models = user_api_key_dict.team_models
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = user_api_key_dict.team_models
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = proxy_model_list

    all_models = _get_models_from_access_groups(
        model_access_groups=model_access_groups, all_models=all_models
    )

    verbose_proxy_logger.debug("ALL TEAM MODELS - {}".format(len(all_models)))
    return all_models


def get_complete_model_list(
    key_models: List[str],
    team_models: List[str],
    proxy_model_list: List[str],
    user_model: Optional[str],
    infer_model_from_keys: Optional[bool],
) -> List[str]:
    """Logic for returning complete model list for a given key + team pair"""

    """
    - If key list is empty -> defer to team list
    - If team list is empty -> defer to proxy model list

    If list contains wildcard -> return known provider models
    """
    unique_models = set()
    if key_models:
        unique_models.update(key_models)
    elif team_models:
        unique_models.update(team_models)
    else:
        unique_models.update(proxy_model_list)

        if user_model:
            unique_models.add(user_model)

        if infer_model_from_keys:
            valid_models = get_valid_models()
            unique_models.update(valid_models)

    models_to_remove = set()
    all_wildcard_models = []
    for model in unique_models:
        if _check_wildcard_routing(model=model):
            provider = model.split("/")[0]
            # get all known provider models
            wildcard_models = get_provider_models(provider=provider)
            if wildcard_models is not None:
                models_to_remove.add(model)
                all_wildcard_models.extend(wildcard_models)

    for model in models_to_remove:
        unique_models.remove(model)

    return list(unique_models) + all_wildcard_models
