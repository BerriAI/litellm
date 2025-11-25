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
    def _get_realtime_protocol(self) -> str:
        """Return the configured realtime protocol.

        Supported values (case-insensitive):
        - "beta"  -> use legacy `/openai/realtime` (current default)
        - "v1"    -> use `/openai/v1/realtime`
        - "ga"    -> alias for "v1" (GA path is v1)

        If the parameter is missing or invalid, we fall back to the current
        behavior for full backwards compatibility.
        """

        # `litellm_params` is the standard place to configure provider-specific
        # behavior. We keep this defensive in case the attribute isn't set.
        params: Any = getattr(self, "litellm_params", None)
        if not isinstance(params, dict):
            return "beta"

        value = params.get("realtime_protocol")
        if not isinstance(value, str):
            return "beta"

        value_normalized = value.lower()
        if value_normalized in {"v1", "ga"}:
            return "v1"

        # Treat anything else (including explicit "beta") as current default
        return "beta"

    def _construct_url(
        self,
        api_base: str,
        model: str,
        api_version: str,
    ) -> str:
        """Construct the websocket URL for Azure OpenAI realtime.

        Example default output (beta / legacy behavior):
        "wss://my-endpoint-sweden-berri992.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview";

        When `realtime_protocol` is set to "v1" or "GA" via `litellm_params`,
        this switches to `/openai/v1/realtime`.
        """

        api_base = api_base.replace("https://", "wss://")

        protocol = self._get_realtime_protocol()
        if protocol == "v1":
            path = "/openai/v1/realtime"
        else:
            # default / beta behavior
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
    ):
        import websockets
        from websockets.asyncio.client import ClientConnection

        if api_base is None:
            raise ValueError("api_base is required for Azure OpenAI calls")
        if api_version is None:
            raise ValueError("api_version is required for Azure OpenAI calls")

        url = self._construct_url(api_base, model, api_version)

        try:
            ssl_context = get_shared_realtime_ssl_context()
            async with websockets.connect(  # type: ignore
                url,
                extra_headers={
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
            pass
