"""
Transformation for Calling Google models in their native format.
"""
from typing import Dict, Literal, Optional, Union

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
        litellm_params: Optional[Union[GenericLiteLLMParams, dict]],
    ) -> dict:
        default_headers = {
            "Content-Type": "application/json",
        }

        if api_key is not None:
            default_headers[self.HEADER_NAME] = f"{self.BEARER_PREFIX} {api_key}"
        if headers is not None:
            default_headers.update(headers)

        return default_headers

    def _camel_to_snake(self, camel_str: str) -> str:
        """Convert camelCase to snake_case"""
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()

    def map_generate_content_optional_params(
        self,
        generate_content_config_dict,
        model: str,
    ):
        """
        Map Google GenAI parameters to provider-specific format.

        Args:
            generate_content_optional_params: Optional parameters for generate content
            model: The model name

        Returns:
            Mapped parameters for the provider
        """
        from litellm.types.google_genai.main import GenerateContentConfigDict

        _generate_content_config_dict = GenerateContentConfigDict()

        for param, value in generate_content_config_dict.items():
            camel_case_key = self._camel_to_snake(param)
            _generate_content_config_dict[camel_case_key] = value
        return dict(_generate_content_config_dict)

    def transform_generate_content_request(
        self,
        model: str,
        contents: any,
        tools: Optional[any],
        generate_content_config_dict: Dict,
        system_instruction: Optional[any] = None,
    ) -> dict:
        """
        Transform the generate content request for Vertex AI.
        Since Vertex AI natively supports Google GenAI format, we can pass most fields directly.
        """
        # Build the request in Google GenAI format that Vertex AI expects
        result = {
            "model": model,
            "contents": contents,
        }

        # Add tools if provided
        if tools:
            result["tools"] = tools

        # Add systemInstruction if provided
        if system_instruction:
            result["systemInstruction"] = system_instruction

        # Handle generationConfig - Vertex AI expects it in the same format
        if generate_content_config_dict:
            result["generationConfig"] = generate_content_config_dict

        return result
