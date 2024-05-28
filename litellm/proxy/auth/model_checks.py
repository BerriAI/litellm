# What is this?
## Common checks for /v1/models and `/model/info`
from typing import List, Optional
from litellm.proxy._types import UserAPIKeyAuth, SpecialModelNames
from litellm.utils import get_valid_models
from litellm._logging import verbose_proxy_logger


def get_key_models(
    user_api_key_dict: UserAPIKeyAuth, proxy_model_list: List[str]
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    """
    all_models = []
    if len(user_api_key_dict.models) > 0:
        all_models = user_api_key_dict.models
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = user_api_key_dict.team_models
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = proxy_model_list

    verbose_proxy_logger.debug("ALL KEY MODELS - {}".format(len(all_models)))
    return all_models


def get_team_models(
    user_api_key_dict: UserAPIKeyAuth, proxy_model_list: List[str]
) -> List[str]:
    """
    Returns:
    - List of model name strings
    - Empty list if no models set
    """
    all_models = []
    if len(user_api_key_dict.team_models) > 0:
        all_models = user_api_key_dict.team_models
        if SpecialModelNames.all_team_models.value in all_models:
            all_models = user_api_key_dict.team_models
        if SpecialModelNames.all_proxy_models.value in all_models:
            all_models = proxy_model_list

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

    return list(unique_models)
