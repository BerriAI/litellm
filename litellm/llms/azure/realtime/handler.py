"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Protocol, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.types.realtime import RealtimeQueryParams

from ....litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ....litellm_core_utils.realtime_errors import (
    close_after_upstream_handshake_refusal,
    realtime_error_event,
)
from ....litellm_core_utils.realtime_streaming import (
    RealTimeStreaming,
    ScopedWebSocket,
    client_sent_openai_beta_realtime_header,
)
from ....llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from ..azure import AzureChatCompletion

# BACKEND_WS_URL = "ws://localhost:8080/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"


async def forward_messages(client_ws: Any, backend_ws: Any):
    import websockets

    try:
        while True:
            message = await backend_ws.recv()
            await client_ws.send_text(message)
    except websockets.exceptions.ConnectionClosed:
        pass


def azure_realtime_requires_ga(model: str) -> bool:
    try:
        azure_model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="azure")
    except Exception:  # noqa: BLE001  # unmapped deployments can select a protocol explicitly
        try:
            openai_model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="openai")
        except Exception:  # noqa: BLE001  # unmapped deployments can select a protocol explicitly
            return False
        openai_entry: Final = openai_model_info.get("provider_specific_entry")
        return openai_entry is not None and openai_entry.get("realtime_ga_only") == 1
    azure_entry: Final = azure_model_info.get("provider_specific_entry")
    return azure_entry is not None and azure_entry.get("realtime_ga_only") == 1


def azure_realtime_protocol_for_client(
    configured_protocol: object,
    *,
    model: str,
    realtime_mode: str,
    query_params: RealtimeQueryParams | None,
    websocket: ScopedWebSocket,
) -> str:
    if azure_realtime_requires_ga(model):
        if isinstance(configured_protocol, str) and configured_protocol.upper() not in ("GA", "V1"):
            raise ValueError(f"{model} requires the Azure OpenAI v1 Realtime API")
        return "GA"
    if realtime_mode == "translation" or (query_params or {}).get("intent") == "transcription":
        return "GA"
    if isinstance(configured_protocol, str) and configured_protocol:
        return configured_protocol
    return "beta" if client_sent_openai_beta_realtime_header(websocket) else "GA"


class _ProxyClientWebSocket(Protocol):
    """Client-facing websocket handle: this path only writes to it after a failed handshake."""

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = ..., reason: str | None = ...) -> None: ...


class AzureOpenAIRealtime(AzureChatCompletion):
    @staticmethod
    def get_auth_headers(api_key: str | None, azure_ad_token: str | None) -> Mapping[str, str]:
        """
        Build the websocket handshake auth headers, preferring a static api-key and falling back to
        an Azure AD (Entra ID) bearer token. Never sends both.
        """
        if api_key:
            return MappingProxyType({"api-key": api_key})
        if azure_ad_token:
            return MappingProxyType({"Authorization": f"Bearer {azure_ad_token}"})
        raise ValueError(
            "Missing Azure credentials for the realtime endpoint. Set an api_key, or configure Azure AD auth "
            "(azure_ad_token, tenant_id/client_id/client_secret, or a managed identity)"
        )

    def _construct_url(
        self,
        api_base: str,
        model: str,
        api_version: str | None,
        realtime_protocol: str | None = None,
        query_params: RealtimeQueryParams | None = None,
        realtime_mode: str = "realtime",
    ) -> str:
        """
        Construct Azure realtime WebSocket URL.

        Args:
            api_base: Azure API base URL (will be converted from https:// to wss://)
            model: Model deployment name
            api_version: Azure API version
            realtime_protocol: Protocol version to use:
                - "GA" or "v1": Uses /openai/v1/realtime (GA path)
                - "beta" or None: Uses /openai/realtime (beta path, default)
            query_params: Extra query params to forward (e.g. intent=transcription).

        Returns:
            WebSocket URL string

        Examples:
            beta/default: "wss://.../openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview"
            GA/v1:        "wss://.../openai/v1/realtime?model=gpt-realtime-deployment"
        """
        from urllib.parse import urlencode

        api_base = api_base.replace("https://", "wss://")

        # Determine path based on realtime_protocol (case-insensitive)
        _is_ga: Final = realtime_protocol is not None and realtime_protocol.upper() in (
            "GA",
            "V1",
        )
        intent: Final = (query_params or {}).get("intent")

        path: Final = (
            "/openai/v1/realtime/translations"
            if realtime_mode == "translation"
            else "/openai/v1/realtime"
            if _is_ga
            else "/openai/realtime"
        )
        base_query_parts: Final = (
            (urlencode((("model", model),)),)
            if realtime_mode == "translation"
            else (
                (urlencode((("model", model),)),)
                if intent != "transcription" and (query_params is None or "model" in query_params)
                else ()
            )
            if _is_ga
            else (urlencode((("api-version", api_version), ("deployment", model))),)
        )

        query_parts: Final = (*base_query_parts, urlencode((("intent", intent),))) if intent else base_query_parts

        qs: Final = "&".join(query_parts)
        return f"{api_base}{path}?{qs}" if qs else f"{api_base}{path}"

    async def async_realtime(
        self,
        model: str,
        websocket: _ProxyClientWebSocket,
        logging_obj: LiteLLMLogging,
        api_base: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        client: object | None = None,
        timeout: float | None = None,
        realtime_protocol: str | None = None,
        query_params: RealtimeQueryParams | None = None,
        user_api_key_dict: object | None = None,
        litellm_metadata: dict | None = None,
        realtime_mode: str = "realtime",
    ):
        import websockets
        from websockets.asyncio.client import ClientConnection

        if api_base is None:
            raise ValueError("api_base is required for Azure OpenAI calls")
        backend_uses_beta_protocol: Final = realtime_protocol is None or realtime_protocol.upper() not in ("GA", "V1")
        if api_version is None and backend_uses_beta_protocol:
            raise ValueError("api_version is required for Azure OpenAI calls")

        url: Final = self._construct_url(
            api_base,
            model,
            api_version,
            realtime_protocol=realtime_protocol,
            query_params=query_params,
            realtime_mode=realtime_mode,
        )

        auth_headers: Final = self.get_auth_headers(api_key=api_key, azure_ad_token=azure_ad_token)

        try:
            ssl_context: Final = get_shared_realtime_ssl_context()
            async with websockets.connect(
                url,
                additional_headers=auth_headers,
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
                ssl=ssl_context,
            ) as backend_ws:
                realtime_streaming: Final = RealTimeStreaming(
                    websocket,
                    cast(ClientConnection, backend_ws),
                    logging_obj,
                    model=model,
                    user_api_key_dict=user_api_key_dict,
                    request_data={"litellm_metadata": litellm_metadata or {}},
                    backend_uses_beta_protocol=backend_uses_beta_protocol,
                    force_transcription_model=(
                        model if (query_params or {}).get("intent") == "transcription" else None
                    ),
                    translation_session=realtime_mode == "translation",
                )
                await realtime_streaming.bidirectional_forward()

        except websockets.exceptions.InvalidStatus as e:
            verbose_proxy_logger.exception("Error in AzureOpenAIRealtime.async_realtime")
            await close_after_upstream_handshake_refusal(websocket, e.response.status_code)
        except Exception:
            verbose_proxy_logger.exception("Error in AzureOpenAIRealtime.async_realtime")
            try:
                await websocket.send_text(realtime_error_event("Internal server error", error_type="server_error"))
            except Exception:  # noqa: BLE001  # best-effort notice: a dead client socket must not skip the close below
                pass
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except Exception:  # noqa: BLE001  # the lower layer may have closed the socket already; closing twice is not an error
                pass
