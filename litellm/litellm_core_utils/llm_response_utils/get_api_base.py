from collections.abc import Mapping
from typing import Final

import litellm
from litellm import verbose_logger

from ...litellm_core_utils.get_llm_provider_logic import (
    declared_authenticating_provider,
    get_llm_provider,
)
from ...types.router import LiteLLM_Params


def _api_base_without_login(provider: str) -> str | None:
    if provider == "github_copilot":
        return litellm.GithubCopilotConfig().api_base_without_login()
    if provider == "chatgpt":
        return litellm.ChatGPTConfig().api_base_without_login()
    return None


def _provider_default_api_base(
    model: str,
    custom_llm_provider: str | None,
    stream: bool,
    extra_params: object = None,
    litellm_params: object = None,
    deployment_model_info: object = None,
) -> str | None:
    if custom_llm_provider == "gemini":
        from litellm.litellm_core_utils.url_utils import ensure_api_route
        from litellm.llms.gemini.common_utils import GeminiModelInfo
        from litellm.responses.bridge_check import should_route_to_interactions_api

        gemini_base: Final = GeminiModelInfo.get_api_base().removesuffix("?alt=sse")
        if should_route_to_interactions_api(
            model=model,
            custom_llm_provider=custom_llm_provider,
            optional_params=extra_params,
            litellm_params=litellm_params,
            deployment_model_info=deployment_model_info,
        ):
            suffix: Final = "?alt=sse" if stream else ""
            return f"{ensure_api_route(gemini_base, 'v1beta/interactions')}{suffix}"
        action: Final = "streamGenerateContent" if stream else "generateContent"
        return ensure_api_route(gemini_base, f"v1beta/models/{model}:{action}")
    if custom_llm_provider == "openai":
        return "https://api.openai.com"
    return None


def get_api_base(model: str, optional_params: dict | LiteLLM_Params) -> str | None:
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
            _optional_params = LiteLLM_Params(model=model, **optional_params)  # convert to pydantic object
    except Exception:
        return None
    # get llm provider

    if _optional_params.api_base is not None:
        return _optional_params.api_base
    extra_params: Final = _optional_params.model_extra
    base_url_alias: Final = extra_params.get("base_url") if extra_params is not None else None
    if isinstance(base_url_alias, str) and base_url_alias:
        return base_url_alias

    if litellm.model_alias_map and model in litellm.model_alias_map:
        model = litellm.model_alias_map[model]
    declared: Final = declared_authenticating_provider(model, _optional_params.custom_llm_provider)
    if declared is not None:
        return _api_base_without_login(declared)
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
        verbose_logger.debug("Error occurred in getting api base - %s", e)
        custom_llm_provider = None
        dynamic_api_base = None

    if dynamic_api_base is not None:
        return dynamic_api_base

    stream: Final[bool] = bool(
        getattr(_optional_params, "stream", False) or (extra_params is not None and extra_params.get("stream") is True)
    )
    litellm_params_map: Final[Mapping[str, object]] = _optional_params.model_dump(exclude_none=True)
    deployment_model_info: Final[object] = litellm_params_map.get("model_info")
    raw_ai_project: Final = extra_params.get("vertex_ai_project") if extra_params is not None else None
    raw_ai_location: Final = extra_params.get("vertex_ai_location") if extra_params is not None else None
    vertex_project: Final = _optional_params.vertex_project or (
        raw_ai_project if isinstance(raw_ai_project, str) and raw_ai_project else None
    )
    vertex_location: Final = _optional_params.vertex_location or (
        raw_ai_location if isinstance(raw_ai_location, str) and raw_ai_location else None
    )

    if vertex_project is not None and "claude" not in model:
        from litellm.llms.vertex_ai.common_utils import get_vertex_base_url, validate_vertex_location
        from litellm.responses.bridge_check import should_route_to_interactions_api

        if should_route_to_interactions_api(
            model=model,
            custom_llm_provider=custom_llm_provider or "vertex_ai",
            optional_params=extra_params,
            litellm_params=litellm_params_map,
            deployment_model_info=deployment_model_info,
        ):
            effective_location: Final = validate_vertex_location(vertex_location or "global")
            vertex_host: Final = get_vertex_base_url(effective_location)
            sse_suffix: Final = "?alt=sse" if stream else ""
            return f"{vertex_host}/v1beta1/projects/{vertex_project}/locations/{effective_location}/interactions{sse_suffix}"

    if vertex_location is not None and vertex_project is not None:
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
        from litellm.types.llms.vertex_ai import VertexPartnerProvider

        if "claude" in model:
            _api_base = VertexBase.create_vertex_url(
                vertex_location=vertex_location,
                vertex_project=vertex_project,
                model=model,
                stream=stream,
                partner=VertexPartnerProvider.claude,
            )
        else:
            if stream:
                _api_base = f"{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:streamGenerateContent"
            else:
                _api_base = f"{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:generateContent"
        return _api_base

    return _provider_default_api_base(
        model,
        custom_llm_provider,
        stream,
        extra_params,
        litellm_params=litellm_params_map,
        deployment_model_info=deployment_model_info,
    )
