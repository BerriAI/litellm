"""
This file contains the calling OpenAI's `/v1/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from typing import Any, Optional, cast

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.types.realtime import RealtimeQueryParams

from ....litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ....litellm_core_utils.realtime_streaming import RealTimeStreaming
from ....llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from ..openai import OpenAIChatCompletion


class OpenAIRealtime(OpenAIChatCompletion):
    """
    Base handler for OpenAI-compatible realtime WebSocket connections.
    
    Subclasses can override template methods to customize:
    - _get_default_api_base(): Default API base URL
    - _get_additional_headers(): Extra headers beyond Authorization
    - _get_ssl_config(): SSL configuration for WebSocket connection
    """
    
    def _get_default_api_base(self) -> str:
        """
        Get the default API base URL for this provider.
        Override this in subclasses to set provider-specific defaults.
        """
        return "https://api.openai.com/"
    
    def _get_additional_headers(self, api_key: str) -> dict:
        """
        Get additional headers beyond Authorization.
        Override this in subclasses to customize headers (e.g., remove OpenAI-Beta).
        
        Args:
            api_key: API key for authentication
            
        Returns:
            Dictionary of additional headers
        """
        return {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
    
    def _get_ssl_config(self, url: str) -> Any:
        """
        Get SSL configuration for WebSocket connection.
        Override this in subclasses to customize SSL behavior.
        
        Args:
            url: WebSocket URL (ws:// or wss://)
            
        Returns:
            SSL configuration (None, True, or SSLContext)
        """
        if url.startswith("ws://"):
            return None
        
        # Use the shared SSL context which respects custom CA certs and SSL settings
        ssl_config = get_shared_realtime_ssl_context()
        
        # If ssl_config is False (ssl_verify=False), websockets library needs True instead
        # to establish connection without verification (False would fail)
        if ssl_config is False:
            return True
        
        return ssl_config
    
    def _construct_url(self, api_base: str, query_params: RealtimeQueryParams) -> str:
        """
        Construct the backend websocket URL with all query parameters (including 'model').
        """
        from httpx import URL

        api_base = api_base.replace("https://", "wss://")
        api_base = api_base.replace("http://", "ws://")
        url = URL(api_base)
        # Set the correct path
        url = url.copy_with(path="/v1/realtime")
        # Include all query parameters including 'model'
        if query_params:
            url = url.copy_with(params=query_params)
        return str(url)

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLogging,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
        query_params: Optional[RealtimeQueryParams] = None,
    ):
        import websockets
        from websockets.asyncio.client import ClientConnection
        
        if api_base is None:
            api_base = self._get_default_api_base()
        if api_key is None:
            raise ValueError("api_key is required for OpenAI realtime calls")

        # Use all query params if provided, else fallback to just model
        if query_params is None:
            query_params = {"model": model}
        url = self._construct_url(api_base, query_params)

        try:
            # Get provider-specific SSL configuration
            ssl_config = self._get_ssl_config(url)
            
            # Get provider-specific headers
            headers = self._get_additional_headers(api_key)
            
            # Log a masked request preview consistent with other endpoints.
            logging_obj.pre_call(
                input=None,
                api_key=api_key,
                additional_args={
                    "api_base": url,
                    "headers": headers,
                    "complete_input_dict": {"query_params": query_params},
                },
            )
            async with websockets.connect(  # type: ignore
                url,
                additional_headers=headers,  # type: ignore
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
                ssl=ssl_config,
            ) as backend_ws:
                realtime_streaming = RealTimeStreaming(
                    websocket, cast(ClientConnection, backend_ws), logging_obj
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
