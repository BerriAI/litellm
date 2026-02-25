"""
OpenAI Responses API WebSocket Mode handler.

Implements the WebSocket transport for OpenAI's Responses API
(wss://api.openai.com/v1/responses).

Protocol summary (from https://developers.openai.com/api/docs/guides/websocket-mode/):
  - Client connects via WebSocket to /v1/responses
  - Client sends `response.create` events; payload mirrors the HTTP Responses
    create body but omits transport-specific fields (`stream`, `background`).
  - Server streams back the same SSE event types used by the HTTP streaming
    endpoint, wrapped in JSON-framed WebSocket messages.
  - Client may continue a conversation by sending another `response.create`
    with `previous_response_id` and incremental input.
  - A warmup request (`generate: false`) can be sent to pre-populate
    connection state without triggering generation.
"""

from typing import Any, Optional

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.responses_websocket_streaming import (
    ResponsesWebSocketStreaming,
)
from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context


class OpenAIResponsesWebSocket:
    """
    Handler for OpenAI Responses API WebSocket connections.

    Mirrors the structure of ``OpenAIRealtime`` but targets the
    ``/v1/responses`` WebSocket endpoint instead of ``/v1/realtime``.
    """

    def _get_default_api_base(self) -> str:
        return "https://api.openai.com/v1"

    def _get_headers(self, api_key: str) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
        }

    def _get_ssl_config(self, url: str) -> Any:
        if url.startswith("ws://"):
            return None
        ssl_config = get_shared_realtime_ssl_context()
        if ssl_config is False:
            return True
        return ssl_config

    def _construct_url(self, api_base: str) -> str:
        from httpx import URL

        api_base = api_base.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        url = URL(api_base)
        if not url.raw_path.endswith(b"/responses"):
            url = url.copy_with(path="/v1/responses")
        return str(url)

    async def async_responses_websocket(
        self,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        user_api_key_dict: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        import websockets

        if api_base is None:
            api_base = self._get_default_api_base()
        if api_key is None:
            raise ValueError("api_key is required for OpenAI Responses WebSocket calls")

        url = self._construct_url(api_base)
        headers = self._get_headers(api_key)
        ssl_config = self._get_ssl_config(url)

        logging_obj.pre_call(
            input=None,
            api_key=api_key,
            additional_args={
                "api_base": url,
                "headers": headers,
                "complete_input_dict": {},
            },
        )

        try:
            async with websockets.connect(  # type: ignore
                url,
                additional_headers=headers,
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

        except websockets.exceptions.InvalidStatusCode as e:  # type: ignore
            await websocket.close(code=e.status_code, reason=str(e))
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
