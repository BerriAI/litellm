"""
Transformation for Calling Google models in their native format.
"""
from typing import Literal, Optional, Union

from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
from litellm.types.router import GenericLiteLLMParams


class VertexAIGoogleGenAIConfig(GoogleGenAIConfig):
    """
    Configuration for calling Google models in their native format.
    """
    HEADER_NAME = "Authorization"
    BEARER_PREFIX = "Bearer"
    
    @property
    def custom_llm_provider(self) -> Literal["gemini", "vertex_ai"]:
        return "vertex_ai"
    

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
            default_headers[self.HEADER_NAME] = f"{self.BEARER_PREFIX} {api_key}"
        if headers is not None:
            default_headers.update(headers)

        return default_headers
    