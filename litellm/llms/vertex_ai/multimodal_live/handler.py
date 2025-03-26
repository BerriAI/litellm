"""
This file contains the calling Azure OpenAI's `/openai/realtime` endpoint.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from typing import Any, Optional

from ....litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from ....litellm_core_utils.gemini_realtime_streaming import GeminiRealTimeStreaming
from ..gemini.vertex_and_google_ai_studio_gemini import VertexLLM, VertexGeminiConfig
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from typing import Literal

class GeminiLive(VertexLLM):

    def __init__(self) -> None:
        super().__init__()

    def _construct_url(self, api_base: str) -> str:
        """
        Example output:
        "BACKEND_WS_URL = "wss://localhost:8080/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"";
        """
        api_base = api_base.replace("https://", "wss://")
        api_base = api_base.replace("http://", "ws://")
        return f"{api_base}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLogging,        
        vertex_location: Optional[str],
        optional_params: dict,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
        voice_name: Optional[str] = "Aoede",
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        vertex_project: Optional[str] = None,
        extra_headers: Optional[dict] = None,

    ):
        import websockets

        if api_base is None:
            raise ValueError("api_base is required for Gemini calls")

        url = self._construct_url(api_base)

        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": voice_name}}
            },
        }

        vertex_location = self.get_vertex_region(vertex_region=vertex_location)

        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        headers = VertexGeminiConfig().validate_environment(
            api_key=_auth_header,
            headers=extra_headers,
            model=model,
            messages=[],
            optional_params=optional_params,
        )

        try:
            async with websockets.connect(  # type: ignore
                url,
                extra_headers=headers,
            ) as backend_ws:
                realtime_streaming = GeminiRealTimeStreaming(
                    websocket, backend_ws, model, config, logging_obj
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
                    # The WebSocket is already closed or the response is completed, so we can
                    # ignore this error
                    pass
                else:
                    # If it's a different RuntimeError, we might want to log it or handle it
                    # differently
                    raise Exception(
                        f"Unexpected error while closing WebSocket: {close_error}"
                    )