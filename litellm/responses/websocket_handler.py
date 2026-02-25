"""
WebSocket handler for the OpenAI Responses API WebSocket mode.

Receives a fully-resolved HTTP URL and auth headers from the entry
point in ``litellm.responses.websocket`` (which uses the same
``BaseResponsesAPIConfig`` credential-resolution as the HTTP path).

This handler only owns the WebSocket-specific concerns:
  - Converting the HTTP URL to a WSS URL
  - Opening the ``websockets`` connection
  - Bidirectional message forwarding via ``ResponsesWebSocketStreaming``
"""

from typing import Any, Optional

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from litellm.responses.websocket_streaming import ResponsesWebSocketStreaming


class OpenAIResponsesWebSocketHandler:
    """
    Handles the WebSocket connection lifecycle for the Responses API.

    Analogous to ``OpenAIRealtime`` but for ``/v1/responses``.
    """

    @staticmethod
    def _http_url_to_ws(http_url: str) -> str:
        return (
            http_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
        )

    @staticmethod
    def _get_ssl_config(url: str) -> Any:
        if url.startswith("ws://"):
            return None
        ssl_config = get_shared_realtime_ssl_context()
        if ssl_config is False:
            return True
        return ssl_config

    async def async_responses_websocket(
        self,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        http_url: str,
        auth_headers: dict,
        user_api_key_dict: Optional[Any] = None,
    ) -> None:
        import websockets

        ws_url = self._http_url_to_ws(http_url)
        ssl_config = self._get_ssl_config(ws_url)

        logging_obj.pre_call(
            input=None,
            api_key="",
            additional_args={
                "api_base": ws_url,
                "headers": auth_headers,
                "complete_input_dict": {},
            },
        )

        try:
            async with websockets.connect(  # type: ignore
                ws_url,
                additional_headers=auth_headers,
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
                ssl=ssl_config,
            ) as backend_ws:
                streaming = ResponsesWebSocketStreaming(
                    websocket=websocket,
                    backend_ws=backend_ws,
                    logging_obj=logging_obj,
                    user_api_key_dict=user_api_key_dict,
                )
                await streaming.bidirectional_forward()

        except Exception as e:
            try:
                await websocket.close(
                    code=1011, reason=f"Internal server error: {str(e)}"
                )
            except RuntimeError as close_error:
                if "already completed" not in str(
                    close_error
                ) and "websocket.close" not in str(close_error):
                    raise Exception(
                        f"Unexpected error while closing WebSocket: {close_error}"
                    )
