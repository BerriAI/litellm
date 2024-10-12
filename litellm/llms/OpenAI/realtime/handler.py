"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

import asyncio
from typing import Any, Optional

from ....litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ....litellm_core_utils.realtime_streaming import RealTimeStreaming
from ..openai import OpenAIChatCompletion


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
        logging_obj: LiteLLMLogging,
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
                realtime_streaming = RealTimeStreaming(
                    websocket, backend_ws, logging_obj
                )
                await realtime_streaming.bidirectional_forward()

        except websockets.exceptions.InvalidStatusCode as e:  # type: ignore
            await websocket.close(code=e.status_code, reason=str(e))
        except Exception as e:
            try:
                await websocket.close(
                    code=1011, reason=f"Internal server error: {str(e)}"
                )
            except RuntimeError as close_error:
                if "already completed" in str(close_error) or "websocket.close" in str(
                    close_error
                ):
                    # The WebSocket is already closed or the response is completed, so we can ignore this error
                    pass
                else:
                    # If it's a different RuntimeError, we might want to log it or handle it differently
                    raise Exception(
                        f"Unexpected error while closing WebSocket: {close_error}"
                    )
