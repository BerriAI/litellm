"""
Transformation for Vertex AI Lyria music generation models.

Lyria uses the Interactions API (v1beta1), NOT generateContent:
  POST https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/interactions
  Body: {"model": "lyria-3-pro-preview", "input": [{"type": "text", "text": "..."}]}
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    HTTPHandler = Any
    AsyncHTTPHandler = Any
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


class LyriaError(BaseLLMException):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message=message, status_code=status_code)


class LyriaConfig(BaseConfig, VertexBase):
    """
    Configuration for Vertex AI Lyria music generation models.

    Uses the Interactions API:
      POST https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/interactions
    """

    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        VertexBase.__init__(self)

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

    def _get_auth_headers(self, litellm_params: dict) -> Dict[str, str]:
        vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params)
        vertex_project = self.safe_get_vertex_ai_project(litellm_params)
        access_token, _ = self.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

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
        headers.update(self._get_auth_headers(litellm_params))
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Strip provider prefix (e.g. "lyria-3-pro-preview" from "vertex_ai/lyria-3-pro-preview")
        lyria_model = model.split("/")[-1] if "/" in model else model
        prompt = convert_content_list_to_str(messages[-1])
        return {
            "model": lyria_model,
            "input": [{"type": "text", "text": prompt}],
        }

    def transform_response(
        self,
        model: str,
        raw_response: Any,
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
        except Exception:
            response_json = raw_response.text

        # Lyria returns a JSON array; wrap as string in message content so callers can parse audio data
        content = (
            json.dumps(response_json)
            if not isinstance(response_json, str)
            else response_json
        )

        model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content=content),
            )
        ]
        model_response.model = model
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return LyriaError(status_code=status_code, message=error_message)
