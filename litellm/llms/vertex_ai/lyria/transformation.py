"""
Transformation for Vertex AI Lyria music generation models.

Lyria uses the Interactions API (v1beta1), not generateContent:
  POST https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/interactions
  Body: {"model": "lyria-3-pro-preview", "input": [{"type": "text", "text": "..."}]}
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    Choices,
    Message,
    ModelResponse,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class LyriaError(BaseLLMException):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message=message, status_code=status_code)


class LyriaConfig(BaseConfig, VertexBase):
    def __init__(self):
        BaseConfig.__init__(self)
        VertexBase.__init__(self)

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """
        The Interactions API rejects a "stream" field in the body. Returning
        False stops the HTTP handler from injecting "stream": true and 400ing.
        """
        return False

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Lyria returns one JSON object with no SSE, so a stream=True caller is
        served the response as a single fake-streamed chunk.
        """
        return stream is True

    def get_supported_openai_params(self, model: str) -> List[str]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        vertex_project = self.safe_get_vertex_ai_project(litellm_params)
        if not vertex_project:
            raise LyriaError(
                status_code=400,
                message="vertex_project is required for Lyria. Set VERTEXAI_PROJECT env var.",
            )
        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{vertex_project}/locations/global/interactions"
        verbose_logger.debug(f"Lyria URL: {url}")
        return url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params)
        vertex_project = self.safe_get_vertex_ai_project(litellm_params)
        access_token, _ = self.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )
        headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        non_system = [m for m in messages if m.get("role") != "system"]
        if len(non_system) > 1:
            verbose_logger.warning(
                "Lyria accepts a single prompt; only the last message will be used. "
                "Multi-turn conversation history is not supported by the Interactions API."
            )
        if not non_system:
            raise LyriaError(
                status_code=400,
                message="Lyria requires at least one non-system message as the music prompt.",
            )
        lyria_model = model.split("/")[-1] if "/" in model else model
        prompt = convert_content_list_to_str(non_system[-1])
        if not prompt.strip():
            raise LyriaError(
                status_code=400,
                message="Lyria prompt is empty. The last non-system message must contain text content.",
            )
        return {
            "model": lyria_model,
            "input": [{"type": "text", "text": prompt}],
        }

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise LyriaError(
                status_code=500,
                message=f"Failed to parse Lyria API response as JSON: {e}. Raw: {raw_response.text[:200]}",
            )

        outputs: list = response_json.get("outputs", [])
        if not outputs:
            raise LyriaError(
                status_code=500,
                message=f"Lyria API returned no outputs. Response: {str(response_json)[:200]}",
            )

        lyrics_text: Optional[str] = None
        caption_text: Optional[str] = None
        audio_b64: Optional[str] = None
        audio_mime_type: str = "audio/mpeg"
        interaction_id: Optional[str] = response_json.get("id")

        # The API emits text outputs in order: the first is the timestamped
        # lyrics, the second is the caption. Audio is keyed by type.
        text_items_seen = 0
        for item in outputs:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "audio":
                audio_b64 = item.get("data")
                audio_mime_type = item.get("mime_type", "audio/mpeg")
            elif item_type == "text":
                text_val = item.get("text", "")
                if text_items_seen == 0:
                    lyrics_text = text_val
                elif text_items_seen == 1:
                    caption_text = text_val
                text_items_seen += 1

        if not audio_b64:
            raise LyriaError(
                status_code=500,
                message=f"Lyria API returned no audio output. Outputs: {str(outputs)[:200]}",
            )

        audio_response = ChatCompletionAudioResponse(
            data=audio_b64,
            expires_at=int(time.time()) + 86400,
            transcript=lyrics_text or "",
        )

        provider_specific: Dict[str, Any] = {
            "audio_mime_type": audio_mime_type,
        }
        if caption_text is not None:
            provider_specific["caption"] = caption_text
        if interaction_id is not None:
            provider_specific["interaction_id"] = interaction_id

        model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    audio=audio_response,
                    provider_specific_fields=provider_specific,
                ),
            )
        ]
        model_response.model = model
        model_response.usage = Usage(  # type: ignore[attr-defined]
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Lyria bills a flat fee per generation, so the price lives in
        # output_cost_per_image and is surfaced through response_cost, which
        # _response_cost_calculator returns verbatim instead of running token math.
        try:
            model_info = litellm.get_model_info(
                model=model, custom_llm_provider="vertex_ai"
            )
            cost_per_generation: float = model_info.get("output_cost_per_image") or 0.0
        except Exception:
            cost_per_generation = 0.0

        model_response._hidden_params = {
            **getattr(model_response, "_hidden_params", {}),
            "response_cost": cost_per_generation,
        }

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return LyriaError(status_code=status_code, message=error_message)
