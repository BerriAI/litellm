"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from typing import Any, Optional, cast

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES

from ....litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ....litellm_core_utils.realtime_streaming import RealTimeStreaming
from ....llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from ..azure import AzureChatCompletion
from litellm._logging import verbose_proxy_logger

# BACKEND_WS_URL = "ws://localhost:8080/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"


async def forward_messages(client_ws: Any, backend_ws: Any):
    import websockets

    try:
        while True:
            message = await backend_ws.recv()
            await client_ws.send_text(message)
    except websockets.exceptions.ConnectionClosed:  # type: ignore
        pass


class AzureOpenAIRealtime(AzureChatCompletion):
    def _construct_url(
        self,
        api_base: str,
        model: str,
        api_version: str,
        realtime_protocol: Optional[str] = None,
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

        Returns:
            WebSocket URL string

        Examples:
            beta/default: "wss://.../openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview"
            GA/v1:        "wss://.../openai/v1/realtime?model=gpt-realtime-deployment"
        """
        api_base = api_base.replace("https://", "wss://")

        # Determine path based on realtime_protocol
        if realtime_protocol in ("GA", "v1"):
            path = "/openai/v1/realtime" 
            return f"{api_base}{path}?model={model}"
        else:
            # Default to beta path for backwards compatibility
            path = "/openai/realtime"
            return f"{api_base}{path}?api-version={api_version}&deployment={model}"

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
        realtime_protocol: Optional[str] = None,
    ):
        import websockets
        from websockets.asyncio.client import ClientConnection

        if api_base is None:
            raise ValueError("api_base is required for Azure OpenAI calls")
        if api_version is None:
            raise ValueError("api_version is required for Azure OpenAI calls")

        url = self._construct_url(
            api_base, model, api_version, realtime_protocol=realtime_protocol
        )

        try:
            ssl_context = get_shared_realtime_ssl_context()
            async with websockets.connect(  # type: ignore
                url,
                additional_headers={
                    "api-key": api_key,  # type: ignore
                },
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
                ssl=ssl_context,
            ) as backend_ws:
                realtime_streaming = RealTimeStreaming(
                    websocket, cast(ClientConnection, backend_ws), logging_obj
                )
                await realtime_streaming.bidirectional_forward()

        except websockets.exceptions.InvalidStatusCode as e:  # type: ignore
            await websocket.close(code=e.status_code, reason=str(e))
        except Exception:
            verbose_proxy_logger.exception("Error in AzureOpenAIRealtime.async_realtime")
            pass
