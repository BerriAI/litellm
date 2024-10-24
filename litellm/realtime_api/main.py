"""Abstraction function for OpenAI's realtime API"""

import os
from typing import Any, Optional

import litellm
from litellm import get_llm_provider
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

from ..litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ..llms.AzureOpenAI.realtime.handler import AzureOpenAIRealtime
from ..llms.OpenAI.realtime.handler import OpenAIRealtime
from ..utils import client as wrapper_client

azure_realtime = AzureOpenAIRealtime()
openai_realtime = OpenAIRealtime()


@wrapper_client
async def _arealtime(
    model: str,
    websocket: Any,  # fastapi websocket
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    azure_ad_token: Optional[str] = None,
    client: Optional[Any] = None,
    timeout: Optional[float] = None,
    **kwargs,
):
    """
    Private function to handle the realtime API call.

    For PROXY use only.
    """
    litellm_logging_obj: LiteLLMLogging = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
    proxy_server_request = kwargs.get("proxy_server_request", None)
    model_info = kwargs.get("model_info", None)
    metadata = kwargs.get("metadata", {})
    user = kwargs.get("user", None)
    litellm_params = GenericLiteLLMParams(**kwargs)

    model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    litellm_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params={},
        litellm_params={
            "litellm_call_id": litellm_call_id,
            "proxy_server_request": proxy_server_request,
            "model_info": model_info,
            "metadata": metadata,
            "preset_cache_key": None,
            "stream_response": {},
        },
        custom_llm_provider=_custom_llm_provider,
    )

    if _custom_llm_provider == "azure":
        api_base = (
            dynamic_api_base
            or litellm_params.api_base
            or litellm.api_base
            or get_secret_str("AZURE_API_BASE")
        )
        # set API KEY
        api_key = (
            dynamic_api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("AZURE_API_KEY")
        )

        await azure_realtime.async_realtime(
            model=model,
            websocket=websocket,
            api_base=api_base,
            api_key=api_key,
            api_version="2024-10-01-preview",
            azure_ad_token=None,
            client=None,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
        )
    elif _custom_llm_provider == "openai":
        api_base = (
            dynamic_api_base
            or litellm_params.api_base
            or litellm.api_base
            or "https://api.openai.com/"
        )
        # set API KEY
        api_key = (
            dynamic_api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )

        await openai_realtime.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            client=None,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
