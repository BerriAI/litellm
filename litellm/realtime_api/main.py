"""Abstraction function for OpenAI's realtime API"""

import os
from typing import Any, Final, cast

import litellm
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES, request_timeout
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.realtime import (
    RealtimeClientSecretRequest,
    RealtimeExpiresAfter,
    RealtimeQueryParams,
    RealtimeSessionConfig,
    RealtimeTranscriptionSessionRequest,
)
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

azure_realtime: Final = AzureOpenAIRealtime()
openai_realtime: Final = OpenAIRealtime()
bedrock_realtime: Final = BedrockRealtime()
xai_realtime: Final = XAIRealtime()
vertex_llm_base: Final = VertexBase()
base_llm_http_handler = BaseLLMHTTPHandler()


def _with_resolved_session_model(session: dict[str, Any], model_name: str) -> dict[str, Any]:
    if "model" not in session:
        return session
    return {**session, "model": model_name}


def _build_litellm_metadata(kwargs: dict) -> dict:
    """Build the litellm_metadata dict for guardrail checking (internal only, not forwarded to provider)."""
    metadata: Final[dict] = {**(kwargs.get("litellm_metadata") or {})}
    guardrails: Final = (kwargs.get("metadata") or {}).get("guardrails") or kwargs.get("guardrails") or []
    if guardrails:
        metadata["guardrails"] = guardrails
    return metadata


def _get_realtime_http_provider_config(
    custom_llm_provider: str,
    dynamic_api_base: str | None,
    dynamic_api_key: str | None,
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

    provider_config: BaseRealtimeHTTPConfig | None = None
    if custom_llm_provider in LlmProviders._member_map_.values():
        provider_config = ProviderConfigManager.get_provider_realtime_http_config(
            model="",
            provider=LlmProviders(custom_llm_provider),
        )

    raw_api_base: Final = dynamic_api_base or litellm_params.api_base
    raw_api_key: Final = dynamic_api_key or litellm_params.api_key

    if provider_config is not None:
        resolved_api_base = provider_config.get_api_base(api_base=raw_api_base)
        resolved_api_key = provider_config.get_api_key(api_key=raw_api_key)
    else:
        # Fallback for providers without a dedicated HTTP config (treated as OpenAI-compatible).
        resolved_api_base = raw_api_base or litellm.api_base or "https://api.openai.com"
        resolved_api_key = (
            raw_api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY") or ""
        )

    return provider_config, resolved_api_base.rstrip("/"), resolved_api_key


@wrapper_client
async def acreate_realtime_client_secret(
    model: str | None = None,
    session: dict[str, Any] | None = None,
    expires_after: dict[str, Any] | None = None,
    timeout: float | None = None,
    **kwargs,
):
    req: Final = RealtimeClientSecretRequest(
        model=model,
        session=RealtimeSessionConfig(**session) if session else None,
        expires_after=RealtimeExpiresAfter(**expires_after) if expires_after else None,
    )
    model_name = (req.session.model if req.session is not None else None) or req.model or "gpt-4o-realtime-preview"
    litellm_logging_obj: Final[LiteLLMLogging] = kwargs.get("litellm_logging_obj")
    litellm_params: Final = GenericLiteLLMParams(**kwargs)

    (
        model_name,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = get_llm_provider(
        model=model_name,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )
    (
        provider_config,
        resolved_api_base,
        resolved_api_key,
    ) = _get_realtime_http_provider_config(
        custom_llm_provider=custom_llm_provider,
        dynamic_api_base=dynamic_api_base,
        dynamic_api_key=dynamic_api_key,
        litellm_params=litellm_params,
    )
    litellm_logging_obj.update_from_kwargs(
        kwargs=kwargs,
        model=model_name,
        optional_params={"expires_after": expires_after, "session": session},
        litellm_params={"api_base": resolved_api_base},
        custom_llm_provider=custom_llm_provider,
    )
    request_data: Final = req.model_dump(exclude_none=True, exclude={"model"})
    if isinstance(request_data.get("session"), dict):
        request_data["session"] = _with_resolved_session_model(request_data["session"], model_name)
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
async def acreate_realtime_transcription_session(
    model: str | None = None,
    transcription_session: dict[str, Any] | None = None,
    timeout: float | None = None,
    **kwargs,
):
    """
    Create an ephemeral transcription session via POST
    /v1/realtime/transcription_sessions.

    ``transcription_session`` is the upstream request body (input_audio_format,
    input_audio_transcription, turn_detection, …). ``model`` is a LiteLLM-only
    routing hint; the provider model lives in
    ``transcription_session.input_audio_transcription.model``.
    """
    req: Final = RealtimeTranscriptionSessionRequest(
        model=model,
        **(transcription_session or {}),
    )
    model_name = req.resolved_model() or "gpt-realtime-whisper"
    litellm_logging_obj: Final[LiteLLMLogging] = kwargs.get("litellm_logging_obj")
    litellm_params: Final = GenericLiteLLMParams(**kwargs)

    (
        model_name,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = get_llm_provider(
        model=model_name,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )
    (
        provider_config,
        resolved_api_base,
        resolved_api_key,
    ) = _get_realtime_http_provider_config(
        custom_llm_provider=custom_llm_provider,
        dynamic_api_base=dynamic_api_base,
        dynamic_api_key=dynamic_api_key,
        litellm_params=litellm_params,
    )
    litellm_logging_obj.update_from_kwargs(
        kwargs=kwargs,
        model=model_name,
        optional_params={"transcription_session": transcription_session},
        litellm_params={"api_base": resolved_api_base},
        custom_llm_provider=custom_llm_provider,
    )
    request_data: Final = req.model_dump(exclude_none=True, exclude={"model"})
    # Ensure the upstream body's input_audio_transcription.model matches the
    # authorized routing model. This prevents a caller from supplying an allowed
    # top-level model for auth while sneaking a different model into the nested
    # transcription config that gets forwarded to the provider.
    if isinstance(request_data.get("input_audio_transcription"), dict):
        request_data["input_audio_transcription"]["model"] = model_name
    return await base_llm_http_handler.async_realtime_transcription_session_handler(
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
    model: str | None = None,
    session: dict[str, Any] | None = None,
    timeout: float | None = None,
    **kwargs,
):
    model_name = model or "gpt-4o-realtime-preview"
    litellm_logging_obj: Final[LiteLLMLogging] = kwargs.get("litellm_logging_obj")
    litellm_params: Final = GenericLiteLLMParams(**kwargs)

    (
        model_name,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = get_llm_provider(
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
    if session is not None:
        session = _with_resolved_session_model(session, model_name)
    litellm_logging_obj.update_from_kwargs(
        kwargs=kwargs,
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
async def _arealtime(
    model: str,
    websocket: Any,  # fastapi websocket
    api_base: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    azure_ad_token: str | None = None,
    client: Any | None = None,
    timeout: float | None = None,
    query_params: RealtimeQueryParams | None = None,
    **kwargs,
):
    """
    Private function to handle the realtime API call.

    For PROXY use only.
    """
    headers = cast(dict | None, kwargs.get("headers"))
    extra_headers: Final = cast(dict | None, kwargs.get("extra_headers"))
    if headers is None:
        headers = {}
    if extra_headers is not None:
        headers.update(extra_headers)
    litellm_logging_obj: Final[LiteLLMLogging] = kwargs.get("litellm_logging_obj")
    user: Final = kwargs.get("user", None)
    litellm_params: Final = GenericLiteLLMParams(**kwargs)

    litellm_params_dict: Final = get_litellm_params(**kwargs)

    model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = get_llm_provider(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )

    # If the client supplied `model` in the URL, ensure it uses the normalized
    # provider model (no proxy aliases). If they omitted it, preserve that shape
    # for transcription-only sessions like OpenAI's `?intent=transcription`.
    if query_params is not None:
        query_params = {**query_params}
        if "model" in query_params:
            query_params["model"] = model

    litellm_logging_obj.update_from_kwargs(
        kwargs=kwargs,
        model=model,
        user=user,
        optional_params={},
        litellm_params=litellm_params_dict,
        custom_llm_provider=_custom_llm_provider,
    )

    provider_config: BaseRealtimeConfig | None = None
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
            query_params=query_params,
        )
    elif _custom_llm_provider == "azure":
        api_base = dynamic_api_base or litellm_params.api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")
        # set API KEY
        api_key = dynamic_api_key or litellm.api_key or litellm.openai_key or get_secret_str("AZURE_API_KEY")

        api_version = api_version or litellm_params.api_version or "2024-10-01-preview"

        realtime_protocol = (
            kwargs.get("realtime_protocol")
            or litellm_params.get("realtime_protocol")
            or os.environ.get("LITELLM_AZURE_REALTIME_PROTOCOL")
        )
        if realtime_protocol is None and (query_params or {}).get("intent") == "transcription":
            realtime_protocol = "GA"
        realtime_protocol = realtime_protocol or "beta"
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
            query_params=query_params,
            user_api_key_dict=kwargs.get("user_api_key_dict"),
            litellm_metadata=_build_litellm_metadata(kwargs),
        )
    elif _custom_llm_provider == "openai":
        api_base = dynamic_api_base or litellm_params.api_base or litellm.api_base or "https://api.openai.com/"
        # set API KEY
        api_key = dynamic_api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")

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
        aws_region_name: Final = kwargs.get("aws_region_name")
        aws_access_key_id: Final = kwargs.get("aws_access_key_id")
        aws_secret_access_key: Final = kwargs.get("aws_secret_access_key")
        aws_session_token: Final = kwargs.get("aws_session_token")
        aws_role_name: Final = kwargs.get("aws_role_name")
        aws_session_name: Final = kwargs.get("aws_session_name")
        aws_profile_name: Final = kwargs.get("aws_profile_name")
        aws_web_identity_token: Final = kwargs.get("aws_web_identity_token")
        aws_sts_endpoint: Final = kwargs.get("aws_sts_endpoint")
        aws_bedrock_runtime_endpoint: Final = kwargs.get("aws_bedrock_runtime_endpoint")
        aws_external_id: Final = kwargs.get("aws_external_id")

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
            dynamic_api_base or litellm_params.api_base or get_secret_str("XAI_API_BASE") or "https://api.x.ai/v1"
        )
        # set API KEY
        api_key = XAIModelInfo.get_api_key(dynamic_api_key, legacy_generic_before_env=True)

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
        vertex_credentials: Final = (
            kwargs.get("vertex_credentials")
            or kwargs.get("vertex_ai_credentials")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )
        vertex_project: Final = (
            kwargs.get("vertex_project")
            or kwargs.get("vertex_ai_project")
            or litellm.vertex_project
            or get_secret_str("VERTEXAI_PROJECT")
        )
        vertex_location: Final = (
            kwargs.get("vertex_location")
            or kwargs.get("vertex_ai_location")
            or litellm.vertex_location
            or get_secret_str("VERTEXAI_LOCATION")
        )

        resolved_location: Final = vertex_llm_base.get_vertex_region(vertex_region=vertex_location, model=model)

        (
            access_token,
            resolved_project,
        ) = await vertex_llm_base._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        vertex_realtime_config: Final = VertexAIRealtimeConfig(
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
            query_params=query_params,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


async def _realtime_health_check(
    model: str,
    custom_llm_provider: str,
    api_key: str | None,
    api_base: str | None = None,
    api_version: str | None = None,
    realtime_protocol: str | None = None,
    model_params: dict | None = None,
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

    url: str | None = None
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
        url = xai_realtime._construct_url(api_base=api_base or "https://api.x.ai/v1", query_params={"model": model})
    elif custom_llm_provider == "vertex_ai":
        vertex_model_params: Final = model_params or {}
        resolved_location: Final = vertex_llm_base.get_vertex_region(
            vertex_region=VertexBase.safe_get_vertex_ai_location(vertex_model_params),
            model=model,
        )
        (
            access_token,
            resolved_project,
        ) = await vertex_llm_base._ensure_access_token_async(
            credentials=VertexBase.safe_get_vertex_ai_credentials(vertex_model_params),
            project_id=VertexBase.safe_get_vertex_ai_project(vertex_model_params),
            custom_llm_provider="vertex_ai",
        )
        vertex_realtime_config: Final = VertexAIRealtimeConfig(
            access_token=access_token,
            project=resolved_project,
            location=resolved_location,
        )
        url = vertex_realtime_config.get_complete_url(api_base=api_base, model=model)
        ssl_context = get_shared_realtime_ssl_context()
        headers: Final = vertex_realtime_config.validate_environment(headers={}, model=model, api_key=None)
        async with websockets.connect(
            url,
            additional_headers=headers,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            ssl=ssl_context,
        ):
            return True
    else:
        raise ValueError(f"Unsupported model: {model}")
    ssl_context = get_shared_realtime_ssl_context()
    async with websockets.connect(
        url,
        additional_headers={
            "api-key": api_key,
        },
        max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
        ssl=ssl_context,
    ):
        return True
