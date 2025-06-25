"""
Transformation for Calling Google models in their native format.
"""
from typing import Literal, Optional, Tuple, Union

import litellm
from litellm.llms.base_llm.google_genai.transformation import (
    BaseGoogleGenAIGenerateContentConfig,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.secret_managers.main import get_secret_str
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
            Complete URL for the API request
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