"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

import asyncio
from typing import Any, Optional

from ..openai import OpenAIChatCompletion


async def forward_messages(client_ws: Any, backend_ws: Any):
    import websockets

    try:
        while True:
            message = await backend_ws.recv()
            await client_ws.send_text(message)
    except websockets.exceptions.ConnectionClosed:  # type: ignore
        pass


class OpenAIRealtime(OpenAIChatCompletion):
    def _construct_url(self, api_base: str, model: str) -> str:
        """
        Example output:
        "BACKEND_WS_URL = "wss://localhost:8080/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"";
        """
        api_base = api_base.replace("https://", "wss://")
        api_base = api_base.replace("http://", "ws://")
        return f"{api_base}/v1/realtime?model={model}"

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        import websockets

        if api_base is None:
            raise ValueError("api_base is required for Azure OpenAI calls")
        if api_key is None:
            raise ValueError("api_key is required for Azure OpenAI calls")

        url = self._construct_url(api_base, model)

        try:
            async with websockets.connect(  # type: ignore
                url,
                extra_headers={
                    "Authorization": f"Bearer {api_key}",  # type: ignore
                    "OpenAI-Beta": "realtime=v1",
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
