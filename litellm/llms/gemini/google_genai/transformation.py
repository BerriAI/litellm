"""
Transformation for Calling Google models in their native format.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import httpx

import litellm
from litellm.llms.base_llm.google_genai.transformation import (
    BaseGoogleGenAIGenerateContentConfig,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.google_genai.main import (
    GenerateContentConfigDict,
    GenerateContentContentListUnionDict,
    GenerateContentRequestDict,
    GenerateContentResponse,
)
from litellm.types.router import GenericLiteLLMParams


class GoogleGenAIConfig(BaseGoogleGenAIGenerateContentConfig, VertexLLM):
    """
    Configuration for calling Google models in their native format.
    """
    @property
    def custom_llm_provider(self) -> Literal["gemini"]:
        return "gemini"
    
    def __init__(self):
        super().__init__()
        VertexLLM.__init__(self)
    
    def get_supported_generate_content_optional_params(self, model: str) -> List[str]:
        """
        Get the list of supported Google GenAI parameters for the model.

        Args:
            model: The model name

        Returns:
            List of supported parameter names
        """
        return [
            "http_options",
            "system_instruction", 
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "max_output_tokens",
            "stop_sequences",
            "response_logprobs",
            "logprobs",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "response_mime_type",
            "response_schema",
            "routing_config",
            "model_selection_config",
            "safety_settings",
            "tools",
            "tool_config",
            "labels",
            "cached_content",
            "response_modalities",
            "media_resolution",
            "speech_config",
            "audio_timestamp",
            "automatic_function_calling",
            "thinking_config"
        ]


    def map_generate_content_optional_params(
        self,
        generate_content_optional_params: Dict[str, Any],
        model: str,
    ) -> GenerateContentConfigDict:
        """
        Map Google GenAI parameters to provider-specific format.

        Args:
            generate_content_optional_params: Optional parameters for generate content
            model: The model name

        Returns:
            Mapped parameters for the provider
        """
        generate_content_config_dict = GenerateContentConfigDict()
        supported_google_genai_params = self.get_supported_generate_content_optional_params(model)
        for param, value in generate_content_optional_params.items():
            if param in supported_google_genai_params:
                generate_content_config_dict[param] = value
        return generate_content_config_dict
    
    def validate_environment(
        self, 
        api_key: Optional[str],
        headers: Optional[dict],
        model: str,
        litellm_params: Optional[Union[GenericLiteLLMParams, dict]]
    ) -> dict:
        default_headers = {
            "Content-Type": "application/json",
        }
        if api_key is not None:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers is not None:
            default_headers.update(headers)

        return default_headers

    def _get_google_ai_studio_api_key(self, litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("api_key", None)
            or litellm_params.pop("gemini_api_key", None)
            or get_secret_str("GEMINI_API_KEY")
            or litellm.api_key
        )


    async def get_auth_token_and_url(
        self,
        api_base: Optional[str],
        model: str,
        litellm_params: dict,
        stream: bool,
    ) -> Tuple[dict, str]:
        """
        Get the complete URL for the request.

        Args:
            api_base: Base API URL
            model: The model name
            litellm_params: LiteLLM parameters

        Returns:
            Tuple of headers and API base
        """

        vertex_credentials = self.get_vertex_ai_credentials(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)

        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=self.custom_llm_provider,
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=self._get_google_ai_studio_api_key(litellm_params),
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=self.custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=True,
        )

        headers = self.validate_environment(
            api_key=auth_header,
            headers=None,
            model=model,
            litellm_params=litellm_params,
        )

        return headers, api_base
    

    def transform_generate_content_request(
        self,
        model: str,
        contents: GenerateContentContentListUnionDict,
        generate_content_config_dict: GenerateContentConfigDict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:

        typed_generate_content_request = GenerateContentRequestDict(
            model=model,
            contents=contents,
            config=generate_content_config_dict,
        )

        request_dict = cast(dict, typed_generate_content_request)

        return request_dict
    
    def transform_generate_content_response(
        self,
        model: str,
        raw_response: httpx.Response,
    ) -> GenerateContentResponse:
        """
        Transform the raw response from the generate content API.

        Args:
            model: The model name
            raw_response: Raw HTTP response

        Returns:
            Transformed response data
        """
        try:
            response = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming generate content response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        
        return GenerateContentResponse(**response)