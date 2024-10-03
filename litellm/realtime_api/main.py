"""Abstraction function for OpenAI's realtime API"""

import os
from typing import Any, Optional

import litellm
from litellm import get_llm_provider
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

from ..llms.AzureOpenAI.realtime.handler import AzureOpenAIRealtime
from ..llms.OpenAI.realtime.handler import OpenAIRealtime

azure_realtime = AzureOpenAIRealtime()
openai_realtime = OpenAIRealtime()


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
    litellm_params = GenericLiteLLMParams(**kwargs)

    model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
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
            api_base=api_base,
            api_key=api_key,
            client=None,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
