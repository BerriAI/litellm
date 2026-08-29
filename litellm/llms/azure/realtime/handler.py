"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, cast

from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.types.realtime import RealtimeQueryParams

from ....litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ....litellm_core_utils.realtime_streaming import RealTimeStreaming
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

        if _is_ga:
            path = "/openai/v1/realtime"
            query_parts = []
            if intent != "transcription" and (query_params is None or "model" in query_params):
                query_parts.append(urlencode({"model": model}))
        else:
            # Default to beta path for backwards compatibility
            path = "/openai/realtime"
            query_parts = [urlencode({"api-version": api_version, "deployment": model})]

        if intent:
            query_parts.append(urlencode({"intent": intent}))

        qs: Final = "&".join(query_parts)
        return f"{api_base}{path}?{qs}" if qs else f"{api_base}{path}"

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        api_base: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        client: Any | None = None,
        timeout: float | None = None,
        realtime_protocol: str | None = None,
        query_params: RealtimeQueryParams | None = None,
        user_api_key_dict: Any | None = None,
        litellm_metadata: dict | None = None,
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
                )
                await realtime_streaming.bidirectional_forward()

        except websockets.exceptions.InvalidStatusCode as e:
            await websocket.close(code=e.status_code, reason=_redact_string(str(e)))
        except Exception:
            verbose_proxy_logger.exception("Error in AzureOpenAIRealtime.async_realtime")
