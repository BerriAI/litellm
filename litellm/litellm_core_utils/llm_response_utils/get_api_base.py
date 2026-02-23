from typing import Optional, Union

import litellm
from litellm import verbose_logger

from ...litellm_core_utils.get_llm_provider_logic import get_llm_provider
from ...types.router import LiteLLM_Params


def get_api_base(
    model: str, optional_params: Union[dict, LiteLLM_Params]
) -> Optional[str]:
    """
    Returns the api base used for calling the model.

    Parameters:
    - model: str - the model passed to litellm.completion()
    - optional_params - the 'litellm_params' in router.completion *OR* additional params passed to litellm.completion - eg. api_base, api_key, etc. See `LiteLLM_Params` - https://github.com/BerriAI/litellm/blob/f09e6ba98d65e035a79f73bc069145002ceafd36/litellm/router.py#L67

    Returns:
    - string (api_base) or None

    Example:
    ```
    from litellm import get_api_base

    get_api_base(model="gemini/gemini-pro")
    ```
    """

    try:
        if isinstance(optional_params, LiteLLM_Params):
            _optional_params = optional_params
        elif "model" in optional_params:
            _optional_params = LiteLLM_Params(**optional_params)
        else:  # prevent needing to copy and pop the dict
            _optional_params = LiteLLM_Params(
                model=model, **optional_params
            )  # convert to pydantic object
    except Exception:
        return None
    # get llm provider

    if _optional_params.api_base is not None:
        return _optional_params.api_base

    if litellm.model_alias_map and model in litellm.model_alias_map:
        model = litellm.model_alias_map[model]
    try:
        (
            model,
            custom_llm_provider,
            dynamic_api_key,
            dynamic_api_base,
        ) = get_llm_provider(
            model=model,
            custom_llm_provider=_optional_params.custom_llm_provider,
            api_base=_optional_params.api_base,
            api_key=_optional_params.api_key,
        )
    except Exception as e:
        verbose_logger.debug("Error occurred in getting api base - {}".format(str(e)))
        custom_llm_provider = None
        dynamic_api_base = None

    if dynamic_api_base is not None:
        return dynamic_api_base

    stream: bool = getattr(optional_params, "stream", False)

    if (
        _optional_params.vertex_location is not None
        and _optional_params.vertex_project is not None
    ):
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
        from litellm.types.llms.vertex_ai import VertexPartnerProvider

        if "claude" in model:
            _api_base = VertexBase.create_vertex_url(
                vertex_location=_optional_params.vertex_location,
                vertex_project=_optional_params.vertex_project,
                model=model,
                stream=stream,
                partner=VertexPartnerProvider.claude,
            )
        else:
            if stream:
                _api_base = "{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent".format(
                    _optional_params.vertex_location,
                    _optional_params.vertex_project,
                    _optional_params.vertex_location,
                    model,
                )
            else:
                _api_base = "{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:generateContent".format(
                    _optional_params.vertex_location,
                    _optional_params.vertex_project,
                    _optional_params.vertex_location,
                    model,
                )
        return _api_base

    if custom_llm_provider is None:
        return None

    if custom_llm_provider == "gemini":
        if stream:
            _api_base = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent".format(
                model
            )
        else:
            _api_base = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent".format(
                model
            )
        return _api_base
    elif custom_llm_provider == "openai":
        _api_base = "https://api.openai.com"
        return _api_base
    return None
