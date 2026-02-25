"""
Entry point for the Responses API WebSocket mode.

Lives alongside ``litellm.responses.main`` and reuses the same
``ProviderConfigManager`` / ``get_llm_provider`` patterns for
credential resolution.

Currently supports OpenAI.  Other providers can be added by
implementing ``get_websocket_url`` / ``get_websocket_headers``
on their ``BaseResponsesAPIConfig`` subclass.
"""

from typing import Any, Optional, cast

from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, client as wrapper_client

from .websocket_handler import OpenAIResponsesWebSocketHandler

openai_responses_ws_handler = OpenAIResponsesWebSocketHandler()


@wrapper_client
async def _aresponses_websocket(
    model: str,
    websocket: Any,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Responses API WebSocket transport.  For proxy use only.

    Follows the same provider-resolution flow as ``litellm.responses.main.responses``:
      1. ``get_llm_provider`` → model, provider, dynamic key/base
      2. ``ProviderConfigManager.get_provider_responses_api_config`` → config
      3. ``config.validate_environment`` → auth headers
      4. ``config.get_complete_url`` → HTTP URL (converted to WSS)
    """
    headers = cast(Optional[dict], kwargs.get("headers"))
    extra_headers = cast(Optional[dict], kwargs.get("extra_headers"))
    if headers is None:
        headers = {}
    if extra_headers is not None:
        headers.update(extra_headers)

    litellm_logging_obj: LiteLLMLogging = kwargs.get("litellm_logging_obj")  # type: ignore
    user = kwargs.get("user", None)
    litellm_params = GenericLiteLLMParams(**kwargs)
    litellm_params_dict = get_litellm_params(**kwargs)

    model, custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    if dynamic_api_key is not None:
        litellm_params.api_key = dynamic_api_key
    if dynamic_api_base is not None:
        litellm_params.api_base = dynamic_api_base

    litellm_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params={},
        litellm_params=litellm_params_dict,
        custom_llm_provider=custom_llm_provider,
    )

    responses_config: Optional[BaseResponsesAPIConfig] = (
        ProviderConfigManager.get_provider_responses_api_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )
    )

    if responses_config is None:
        raise ValueError(
            f"Responses API WebSocket mode is not supported for provider: "
            f"{custom_llm_provider}. No responses config found."
        )

    auth_headers = responses_config.validate_environment(
        headers={}, model=model, litellm_params=litellm_params
    )

    http_url = responses_config.get_complete_url(
        api_base=litellm_params.api_base,
        litellm_params=litellm_params_dict,
    )

    await openai_responses_ws_handler.async_responses_websocket(
        websocket=websocket,
        logging_obj=litellm_logging_obj,
        http_url=http_url,
        auth_headers=auth_headers,
        user_api_key_dict=kwargs.get("user_api_key_dict"),
    )
