"""
Entry point for the Responses API WebSocket mode.

Analogous to ``litellm.realtime_api.main._arealtime`` but for the
``/v1/responses`` WebSocket transport.

Currently supports OpenAI only.  Other providers can be added as they ship
their own WebSocket modes.
"""

from typing import Any, Optional, cast

import litellm
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.openai.responses.websocket_handler import OpenAIResponsesWebSocket
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client as wrapper_client

openai_responses_ws = OpenAIResponsesWebSocket()


@wrapper_client
async def _aresponses_websocket(
    model: str,
    websocket: Any,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Private function for the Responses API WebSocket transport.

    For PROXY use only.
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

    model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    litellm_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params={},
        litellm_params=litellm_params_dict,
        custom_llm_provider=_custom_llm_provider,
    )

    if _custom_llm_provider == "openai":
        resolved_api_base = (
            dynamic_api_base
            or litellm_params.api_base
            or litellm.api_base
            or "https://api.openai.com/v1"
        )
        resolved_api_key = (
            dynamic_api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )

        await openai_responses_ws.async_responses_websocket(
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            user_api_key_dict=kwargs.get("user_api_key_dict"),
        )
    else:
        raise ValueError(
            f"Responses API WebSocket mode is not supported for provider: {_custom_llm_provider}. "
            "Currently only 'openai' is supported."
        )
