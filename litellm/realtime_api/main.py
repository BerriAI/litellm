"""Abstraction function for OpenAI's realtime API"""

from typing import Any, Optional, cast

import litellm
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.realtime import RealtimeQueryParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from ..litellm_core_utils.get_litellm_params import get_litellm_params
from ..litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ..llms.azure.realtime.handler import AzureOpenAIRealtime
from ..llms.bedrock.realtime.handler import BedrockRealtime
from ..llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from ..llms.openai.realtime.handler import OpenAIRealtime
from ..llms.xai.realtime.handler import XAIRealtime
from ..utils import client as wrapper_client

azure_realtime = AzureOpenAIRealtime()
openai_realtime = OpenAIRealtime()
bedrock_realtime = BedrockRealtime()
xai_realtime = XAIRealtime()
base_llm_http_handler = BaseLLMHTTPHandler()


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
    query_params: Optional[RealtimeQueryParams] = None,
    **kwargs,
):
    """
    Private function to handle the realtime API call.

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

    # Ensure query params use the normalized provider model (no proxy aliases).
    if query_params is not None:
        query_params = {**query_params, "model": model}

    litellm_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params={},
        litellm_params=litellm_params_dict,
        custom_llm_provider=_custom_llm_provider,
    )

    provider_config: Optional[BaseRealtimeConfig] = None
    if _custom_llm_provider in LlmProviders._member_map_.values():
        provider_config = ProviderConfigManager.get_provider_realtime_config(
            model=model,
            provider=LlmProviders(_custom_llm_provider),
        )
    if provider_config is not None:
        await base_llm_http_handler.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            provider_config=provider_config,
            api_base=api_base,
            api_key=api_key,
            client=client,
            timeout=timeout,
            headers=headers,
        )
    elif _custom_llm_provider == "azure":
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

        api_version = (
            api_version
            or litellm_params.api_version
            or "2024-10-01-preview"
        )
        
        realtime_protocol = (
            kwargs.get("realtime_protocol")
            or "beta"
        )
        await azure_realtime.async_realtime(
            model=model,
            websocket=websocket,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=None,
            client=None,
            timeout=timeout,
            logging_obj=litellm_logging_obj,
            realtime_protocol=realtime_protocol,
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
            query_params=query_params,
        )
    elif _custom_llm_provider == "bedrock":
        # Extract AWS parameters from kwargs
        aws_region_name = kwargs.get("aws_region_name")
        aws_access_key_id = kwargs.get("aws_access_key_id")
        aws_secret_access_key = kwargs.get("aws_secret_access_key")
        aws_session_token = kwargs.get("aws_session_token")
        aws_role_name = kwargs.get("aws_role_name")
        aws_session_name = kwargs.get("aws_session_name")
        aws_profile_name = kwargs.get("aws_profile_name")
        aws_web_identity_token = kwargs.get("aws_web_identity_token")
        aws_sts_endpoint = kwargs.get("aws_sts_endpoint")
        aws_bedrock_runtime_endpoint = kwargs.get("aws_bedrock_runtime_endpoint")
        aws_external_id = kwargs.get("aws_external_id")

        await bedrock_realtime.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            api_base=dynamic_api_base or api_base,
            api_key=dynamic_api_key or api_key,
            timeout=timeout,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_role_name=aws_role_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_external_id=aws_external_id,
        )
    elif _custom_llm_provider == "xai":
        api_base = (
            dynamic_api_base
            or litellm_params.api_base
            or get_secret_str("XAI_API_BASE")
            or "https://api.x.ai/v1"
        )
        # set API KEY
        api_key = (
            dynamic_api_key
            or litellm.api_key
            or get_secret_str("XAI_API_KEY")
        )

        await xai_realtime.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            client=None,
            timeout=timeout,
            query_params=query_params,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


async def _realtime_health_check(
    model: str,
    custom_llm_provider: str,
    api_key: Optional[str],
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    realtime_protocol: Optional[str] = None,
):
    """
    Health check for realtime API - tries connection to the realtime API websocket

    Args:
        model: str - model name
        api_base: str - api base
        api_version: Optional[str] - api version
        api_key: str - api key
        custom_llm_provider: str - custom llm provider
        realtime_protocol: Optional[str] - protocol version ("GA"/"v1" for GA path, "beta"/None for beta path)

    Returns:
        bool - True if connection is successful, False otherwise
    Raises:
        Exception - if the connection is not successful
    """
    import websockets

    url: Optional[str] = None
    if custom_llm_provider == "azure":
        url = azure_realtime._construct_url(
            api_base=api_base or "",
            model=model,
            api_version=api_version or "2024-10-01-preview",
            realtime_protocol=realtime_protocol,
        )
    elif custom_llm_provider == "openai":
        url = openai_realtime._construct_url(
            api_base=api_base or "https://api.openai.com/", query_params={"model": model}
        )
    elif custom_llm_provider == "xai":
        url = xai_realtime._construct_url(
            api_base=api_base or "https://api.x.ai/v1", query_params={"model": model}
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    ssl_context = get_shared_realtime_ssl_context()
    async with websockets.connect(  # type: ignore
        url,
        additional_headers={
            "api-key": api_key,  # type: ignore
        },
        max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
        ssl=ssl_context,
    ):
        return True
