"""Abstraction function for OpenAI's realtime API"""

import os
from typing import Any, Dict, Optional, cast

import litellm
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES, request_timeout
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.realtime import RealtimeClientSecretRequest, RealtimeQueryParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from ..litellm_core_utils.get_litellm_params import get_litellm_params
from ..litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ..llms.azure.realtime.handler import AzureOpenAIRealtime
from ..llms.bedrock.realtime.handler import BedrockRealtime
from ..llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from ..llms.openai.realtime.handler import OpenAIRealtime
from ..llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
from ..llms.vertex_ai.vertex_llm_base import VertexBase
from ..llms.xai.realtime.handler import XAIRealtime
from ..utils import client as wrapper_client

azure_realtime = AzureOpenAIRealtime()
openai_realtime = OpenAIRealtime()
bedrock_realtime = BedrockRealtime()
xai_realtime = XAIRealtime()
vertex_llm_base = VertexBase()
base_llm_http_handler = BaseLLMHTTPHandler()


def _build_litellm_metadata(kwargs: dict) -> dict:
    """Build the litellm_metadata dict for guardrail checking (internal only, not forwarded to provider)."""
    metadata: dict = {**(kwargs.get("litellm_metadata") or {})}
    guardrails = (
        (kwargs.get("metadata") or {}).get("guardrails")
        or kwargs.get("guardrails")
        or []
    )
    if guardrails:
        metadata["guardrails"] = guardrails
    return metadata


def _get_realtime_http_provider_config(
    custom_llm_provider: str,
    dynamic_api_base: Optional[str],
    dynamic_api_key: Optional[str],
    litellm_params: GenericLiteLLMParams,
) -> tuple[Any, str, str]:
    """
    Return (provider_config, resolved_api_base, resolved_api_key) for the
    realtime HTTP endpoints (client_secrets / realtime_calls).

    Uses ProviderConfigManager so each provider keeps its credential-resolution
    and URL-construction logic in its own transformation class.
    """
    from litellm.llms.base_llm.realtime.http_transformation import (
        BaseRealtimeHTTPConfig,
    )

    provider_config: Optional[BaseRealtimeHTTPConfig] = None
    if custom_llm_provider in LlmProviders._member_map_.values():
        provider_config = ProviderConfigManager.get_provider_realtime_http_config(
            model="",
            provider=LlmProviders(custom_llm_provider),
        )

    raw_api_base = dynamic_api_base or litellm_params.api_base
    raw_api_key = dynamic_api_key or litellm_params.api_key

    if provider_config is not None:
        resolved_api_base = provider_config.get_api_base(api_base=raw_api_base)
        resolved_api_key = provider_config.get_api_key(api_key=raw_api_key)
    else:
        # Fallback for providers without a dedicated HTTP config (treated as OpenAI-compatible).
        resolved_api_base = (
            raw_api_base
            or litellm.api_base
            or "https://api.openai.com"
        )
        resolved_api_key = (
            raw_api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
            or ""
        )

    return provider_config, resolved_api_base.rstrip("/"), resolved_api_key


@wrapper_client
async def acreate_realtime_client_secret(
    model: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    expires_after: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    **kwargs,
):
    req = RealtimeClientSecretRequest(
        model=model,
        session=session,
        expires_after=expires_after,
    )
    model_name = (
        (req.session.model if req.session is not None else None)
        or req.model
        or "gpt-4o-realtime-preview"
    )
    litellm_logging_obj: LiteLLMLogging = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_params = GenericLiteLLMParams(**kwargs)

    model_name, custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model_name,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )
    provider_config, resolved_api_base, resolved_api_key = _get_realtime_http_provider_config(
        custom_llm_provider=custom_llm_provider,
        dynamic_api_base=dynamic_api_base,
        dynamic_api_key=dynamic_api_key,
        litellm_params=litellm_params,
    )
    litellm_logging_obj.update_environment_variables(
        model=model_name,
        optional_params={"expires_after": expires_after, "session": session},
        litellm_params={"api_base": resolved_api_base},
        custom_llm_provider=custom_llm_provider,
    )
    request_data = req.model_dump(exclude_none=True, exclude={"model"})
    return await base_llm_http_handler.async_realtime_client_secret_handler(
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        request_data=request_data,
        logging_obj=litellm_logging_obj,
        timeout=timeout or request_timeout,
        provider_config=provider_config,
        model=model_name,
        extra_headers=kwargs.get("extra_headers"),
        client=kwargs.get("client"),
        api_version=litellm_params.api_version,
    )


@wrapper_client
async def arealtime_calls(
    openai_ephemeral_key: str,
    sdp_body: bytes,
    model: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    **kwargs,
):
    model_name = model or "gpt-4o-realtime-preview"
    litellm_logging_obj: LiteLLMLogging = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_params = GenericLiteLLMParams(**kwargs)

    model_name, custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model_name,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )
    provider_config, resolved_api_base, _ = _get_realtime_http_provider_config(
        custom_llm_provider=custom_llm_provider,
        dynamic_api_base=dynamic_api_base,
        dynamic_api_key=dynamic_api_key,
        litellm_params=litellm_params,
    )
    litellm_logging_obj.update_environment_variables(
        model=model_name,
        optional_params={"realtime_calls": True, "session": session},
        litellm_params={"api_base": resolved_api_base},
        custom_llm_provider=custom_llm_provider,
    )
    return await base_llm_http_handler.async_realtime_calls_handler(
        api_base=resolved_api_base,
        openai_ephemeral_key=openai_ephemeral_key,
        sdp_body=sdp_body,
        logging_obj=litellm_logging_obj,
        timeout=timeout or request_timeout,
        provider_config=provider_config,
        model=model_name,
        session_config=session,
        extra_headers=kwargs.get("extra_headers"),
        client=kwargs.get("client"),
        api_version=litellm_params.api_version,
    )


@wrapper_client
async def _arealtime(  # noqa: PLR0915
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
            user_api_key_dict=kwargs.get("user_api_key_dict"),
            litellm_metadata=_build_litellm_metadata(kwargs),
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

        api_version = api_version or litellm_params.api_version or "2024-10-01-preview"

        realtime_protocol = (
            kwargs.get("realtime_protocol")
            or litellm_params.get("realtime_protocol")
            or os.environ.get("LITELLM_AZURE_REALTIME_PROTOCOL")
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
            user_api_key_dict=kwargs.get("user_api_key_dict"),
            litellm_metadata=_build_litellm_metadata(kwargs),
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
            user_api_key_dict=kwargs.get("user_api_key_dict"),
            litellm_metadata=_build_litellm_metadata(kwargs),
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
        api_key = dynamic_api_key or litellm.api_key or get_secret_str("XAI_API_KEY")

        await xai_realtime.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            api_base=api_base,
            api_key=api_key,
            client=None,
            timeout=timeout,
            query_params=query_params,
            user_api_key_dict=kwargs.get("user_api_key_dict"),
            litellm_metadata=_build_litellm_metadata(kwargs),
        )
    elif _custom_llm_provider == "vertex_ai":
        vertex_credentials = (
            kwargs.get("vertex_credentials")
            or kwargs.get("vertex_ai_credentials")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )
        vertex_project = (
            kwargs.get("vertex_project")
            or kwargs.get("vertex_ai_project")
            or litellm.vertex_project
            or get_secret_str("VERTEXAI_PROJECT")
        )
        vertex_location = (
            kwargs.get("vertex_location")
            or kwargs.get("vertex_ai_location")
            or litellm.vertex_location
            or get_secret_str("VERTEXAI_LOCATION")
        )

        resolved_location = vertex_llm_base.get_vertex_region(
            vertex_region=vertex_location, model=model
        )

        (
            access_token,
            resolved_project,
        ) = await vertex_llm_base._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        vertex_realtime_config = VertexAIRealtimeConfig(
            access_token=access_token,
            project=resolved_project,
            location=resolved_location,
        )

        await base_llm_http_handler.async_realtime(
            model=model,
            websocket=websocket,
            logging_obj=litellm_logging_obj,
            provider_config=vertex_realtime_config,
            api_base=dynamic_api_base or litellm_params.api_base,
            api_key=None,
            client=client,
            timeout=timeout,
            headers=headers,
            user_api_key_dict=kwargs.get("user_api_key_dict"),
            litellm_metadata=_build_litellm_metadata(kwargs),
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
            api_base=api_base or "https://api.openai.com/",
            query_params={"model": model},
        )
    elif custom_llm_provider == "xai":
        url = xai_realtime._construct_url(
            api_base=api_base or "https://api.x.ai/v1", query_params={"model": model}
        )
    elif custom_llm_provider == "vertex_ai":
        vertex_location = litellm.vertex_location or get_secret_str("VERTEXAI_LOCATION")
        resolved_location = vertex_llm_base.get_vertex_region(
            vertex_region=vertex_location, model=model
        )
        (
            access_token,
            resolved_project,
        ) = await vertex_llm_base._ensure_access_token_async(
            credentials=None,
            project_id=litellm.vertex_project or get_secret_str("VERTEXAI_PROJECT"),
            custom_llm_provider="vertex_ai",
        )
        vertex_realtime_config = VertexAIRealtimeConfig(
            access_token=access_token,
            project=resolved_project,
            location=resolved_location,
        )
        url = vertex_realtime_config.get_complete_url(api_base=api_base, model=model)
        ssl_context = get_shared_realtime_ssl_context()
        headers = vertex_realtime_config.validate_environment(
            headers={}, model=model, api_key=None
        )
        async with websockets.connect(  # type: ignore
            url,
            additional_headers=headers,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            ssl=ssl_context,
        ):
            return True
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
