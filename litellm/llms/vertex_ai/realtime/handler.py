"""
Vertex AI Realtime API handler.

This module provides the handler for Vertex AI's Realtime (Live) API,
enabling it to work through LiteLLM's unified realtime interface.
"""

import asyncio
from typing import Any, Optional

from litellm._logging import verbose_logger
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.utils import get_secret

class VertexAIRealtime(VertexLLM):
    """
    Handler for Vertex AI Realtime (Live) API.
    
    This class extends the base VertexLLM to provide realtime functionality
    through LiteLLM's unified realtime interface.
    """

    def __init__(self):
        super().__init__()
        self.realtime_config = VertexAIRealtimeConfig()

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLoggingObj,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
        query_params: Optional[dict] = None,
    ):
        """
        Handle realtime WebSocket connection for Vertex AI Live API.
        
        Args:
            model: Model name (e.g., "gemini-2.0-flash-live-preview-04-09")
            websocket: WebSocket connection object
            logging_obj: Logging object
            api_base: Base URL (not used for Vertex AI)
            api_key: API key (not used for Vertex AI)
            client: HTTP client (not used for WebSocket)
            timeout: Connection timeout
            query_params: Query parameters
        """
        import websockets
        from websockets.asyncio.client import ClientConnection

        # Get the complete URL for Vertex AI Live API
        url = self.realtime_config.get_complete_url(api_base, model, api_key)
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add project and location from query params or environment
        if query_params:
            if "vertex_project" in query_params:
                headers["x-goog-user-project"] = query_params["vertex_project"]
            if "vertex_location" in query_params:
                headers["x-goog-vertex-location"] = query_params["vertex_location"]
        
        vertex_credentials = get_secret("VERTEXAI_CREDENTIALS")
    
        # Validate environment and get proper headers
        try:
            headers = self.realtime_config.validate_environment(
                headers=headers,
                model=model,
                api_key=api_key,
                vertex_credentials=vertex_credentials,
            )
            print(f"🔥Headers: {headers}")
        except Exception as e:
            await websocket.close(code=400, reason=f"Authentication error: {str(e)}")
            return
        url = "wss://kimberli-platykurtic-sherrill.ngrok-free.dev"
        try:
            # Connect to Vertex AI Live API WebSocket
            print(f"🔥Connecting to Vertex AI Live API WebSocket: {url}")
            async with websockets.connect(
                url,
                extra_headers=headers,
                timeout=timeout or 30.0,
            ) as backend_ws:
                # Create realtime streaming handler
                from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
                
                realtime_streaming = RealTimeStreaming(
                    websocket,
                    backend_ws,
                    logging_obj,
                    self.realtime_config,
                    model,
                )
                
                # Start bidirectional forwarding
                await realtime_streaming.bidirectional_forward()

        except websockets.exceptions.InvalidStatusCode as e:
            await websocket.close(code=e.status_code, reason=str(e))
        except asyncio.TimeoutError:
            await websocket.close(code=408, reason="Connection timeout")
        except Exception as e:
            try:
                await websocket.close(
                    code=1011, reason=f"Internal server error: {str(e)}"
                )
            except RuntimeError as close_error:
                if "already completed" not in str(close_error) and "websocket.close" not in str(close_error):
                    raise Exception(f"Unexpected error while closing WebSocket: {close_error}")

    def get_realtime_config(self) -> VertexAIRealtimeConfig:
        """
        Get the realtime configuration for this handler.
        
        Returns:
            VertexAIRealtimeConfig: The realtime configuration
        """
        return self.realtime_config
