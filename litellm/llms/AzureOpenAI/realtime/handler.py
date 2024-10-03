"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

import asyncio
from typing import Any, Optional

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
    def _construct_url(self, api_base: str, model: str, api_version: str) -> str:
        """
        Example output:
        "wss://my-endpoint-sweden-berri992.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview";

        """
        api_base = api_base.replace("https://", "wss://")
        return (
            f"{api_base}/openai/realtime?api-version={api_version}&deployment={model}"
        )

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        import websockets

        if api_base is None:
            raise ValueError("api_base is required for Azure OpenAI calls")
        if api_version is None:
            raise ValueError("api_version is required for Azure OpenAI calls")

        url = self._construct_url(api_base, model, api_version)

        try:
            async with websockets.connect(  # type: ignore
                url,
                extra_headers={
                    "api-key": api_key,  # type: ignore
                },
            ) as backend_ws:
                forward_task = asyncio.create_task(
                    forward_messages(websocket, backend_ws)
                )

                try:
                    while True:
                        message = await websocket.receive_text()
                        await backend_ws.send(message)
                except websockets.exceptions.ConnectionClosed:  # type: ignore
                    forward_task.cancel()
                finally:
                    if not forward_task.done():
                        forward_task.cancel()
                        try:
                            await forward_task
                        except asyncio.CancelledError:
                            pass

        except websockets.exceptions.InvalidStatusCode as e:  # type: ignore
            await websocket.close(code=e.status_code, reason=str(e))
        except Exception as e:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
