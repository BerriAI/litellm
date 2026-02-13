import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

import litellm
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexAIGeminiImageGenerationConfig(BaseImageGenerationConfig, VertexLLM):
    """
    Vertex AI Gemini Image Generation Configuration
    
    Uses generateContent API for Gemini image generation models on Vertex AI
    Supports models like gemini-2.5-flash-image, gemini-3-pro-image-preview, etc.
    """
    
    def __init__(self) -> None:
        BaseImageGenerationConfig.__init__(self)
        VertexLLM.__init__(self)
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Gemini image generation supported parameters
        """
        return [
            "n",
            "size",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        mapped_params = {}
        
        for k, v in non_default_params.items():
            if k not in optional_params.keys():
                if k in supported_params:
                    # Map OpenAI parameters to Gemini format
                    if k == "n":
                        mapped_params["candidate_count"] = v
                    elif k == "size":
                        # Map OpenAI size format to Gemini aspectRatio
                        mapped_params["aspectRatio"] = self._map_size_to_aspect_ratio(v)
                    else:
                        mapped_params[k] = v
        
        return mapped_params
    
    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """
        Map OpenAI size format to Gemini aspect ratio format
        """
        aspect_ratio_map = {
            "1024x1024": "1:1",
            "1792x1024": "16:9", 
            "1024x1792": "9:16",
            "1280x896": "4:3",
            "896x1280": "3:4"
        }
        return aspect_ratio_map.get(size, "1:1")
    
    def _resolve_vertex_project(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_project", None)
            or os.environ.get("VERTEXAI_PROJECT")
            or getattr(litellm, "vertex_project", None)
            or get_secret_str("VERTEXAI_PROJECT")
        )

    def _resolve_vertex_location(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_location", None)
            or os.environ.get("VERTEXAI_LOCATION")
            or os.environ.get("VERTEX_LOCATION")
            or getattr(litellm, "vertex_location", None)
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )

    def _resolve_vertex_credentials(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_credentials", None)
            or os.environ.get("VERTEXAI_CREDENTIALS")
            or getattr(litellm, "vertex_credentials", None)
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for Vertex AI Gemini generateContent API
        """
        # Use the model name as provided, handling vertex_ai prefix
        model_name = model
        if model.startswith("vertex_ai/"):
            model_name = model.replace("vertex_ai/", "")

        # If a custom api_base is provided, use it directly
        # This allows users to use proxies or mock endpoints
        if api_base:
            return api_base.rstrip("/")

        # First check litellm_params (where vertex_ai_project/vertex_ai_location are passed)
        # then fall back to environment variables and other sources
        vertex_project = self.safe_get_vertex_ai_project(litellm_params) or self._resolve_vertex_project()
        vertex_location = self.safe_get_vertex_ai_location(litellm_params) or self._resolve_vertex_location()

        if not vertex_project or not vertex_location:
            raise ValueError("vertex_project and vertex_location are required for Vertex AI")

        base_url = get_vertex_base_url(vertex_location)

        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model_name}:generateContent"

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
        headers = headers or {}
        
        # If a custom api_base is provided, skip credential validation
        # This allows users to use proxies or mock endpoints without needing Vertex AI credentials
        _api_base = litellm_params.get("api_base") or api_base
        if _api_base is not None:
            return headers
        
        # First check litellm_params (where vertex_ai_project/vertex_ai_credentials are passed)
        # then fall back to environment variables and other sources
        vertex_project = self.safe_get_vertex_ai_project(litellm_params) or self._resolve_vertex_project()
        vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params) or self._resolve_vertex_credentials()
        access_token, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        return self.set_headers(access_token, headers)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to Gemini format
        
        Uses generateContent API with responseModalities: ["IMAGE"]
        """
        # Prepare messages with the prompt
        contents = [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
        
        # Prepare generation config
        generation_config: Dict[str, Any] = {
            "responseModalities": ["IMAGE"]
        }
        
        # Handle image-specific config parameters
        image_config: Dict[str, Any] = {}
        
        # Map aspectRatio
        if "aspectRatio" in optional_params:
            image_config["aspectRatio"] = optional_params["aspectRatio"]
        elif "aspect_ratio" in optional_params:
            image_config["aspectRatio"] = optional_params["aspect_ratio"]
        
        # Map imageSize (for Gemini 3 Pro)
        if "imageSize" in optional_params:
            image_config["imageSize"] = optional_params["imageSize"]
        elif "image_size" in optional_params:
            image_config["imageSize"] = optional_params["image_size"]
        
        if image_config:
            generation_config["imageConfig"] = image_config
        
        # Handle candidate_count (n parameter)
        if "candidate_count" in optional_params:
            generation_config["candidateCount"] = optional_params["candidate_count"]
        elif "n" in optional_params:
            generation_config["candidateCount"] = optional_params["n"]
        
        request_body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        return request_body

    def _transform_image_usage(self, usage: dict) -> ImageUsage:
        input_tokens_details = ImageUsageInputTokensDetails(
            image_tokens=0,
            text_tokens=0,
        )
        tokens_details = usage.get("promptTokensDetails", [])
        for details in tokens_details:
            if isinstance(details, dict) and (modality := details.get("modality")):
                token_count = details.get("tokenCount", 0)
                if modality == "TEXT":
                    input_tokens_details.text_tokens += token_count
                elif modality == "IMAGE":
                    input_tokens_details.image_tokens += token_count

        return ImageUsage(
            input_tokens=usage.get("promptTokenCount", 0),
            input_tokens_details=input_tokens_details,
            output_tokens=usage.get("candidatesTokenCount", 0),
            total_tokens=usage.get("totalTokenCount", 0),
        )

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform Gemini image generation response to litellm ImageResponse format
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        
        if not model_response.data:
            model_response.data = []

        # Gemini image generation models return in candidates format
        candidates = response_data.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                # Look for inlineData with image
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    if "data" in inline_data:
                        thought_sig = part.get("thoughtSignature")
                        model_response.data.append(ImageObject(
                            b64_json=inline_data["data"],
                            url=None,
                            provider_specific_fields={"thought_signature": thought_sig} if thought_sig else None,
                        ))

        if usage_metadata := response_data.get("usageMetadata", None):
            model_response.usage = self._transform_image_usage(usage_metadata)
        
        return model_response

