"""
Transformation for Calling Google models in their native format.
"""

from typing import Any, Final, Literal

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
        api_key: str | None,
        headers: dict | None,
        model: str,
        litellm_params: GenericLiteLLMParams | dict | None,
    ) -> dict:
        default_headers: Final = {
            "Content-Type": "application/json",
        }

        if api_key is not None:
            default_headers[self.HEADER_NAME] = f"{self.BEARER_PREFIX} {api_key}"
        if headers is not None:
            default_headers.update(headers)

        return default_headers

    def map_generate_content_optional_params(
        self,
        generate_content_config_dict: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        return super().map_generate_content_optional_params(
            generate_content_config_dict=generate_content_config_dict,
            model=model,
        )

    def transform_generate_content_request(
        self,
        model: str,
        contents: Any,
        tools: Any | None,
        generate_content_config_dict: dict,
        system_instruction: Any | None = None,
    ) -> dict:
        """
        Transform the generate content request for Vertex AI.
        Since Vertex AI natively supports Google GenAI format, we can pass most fields directly.
        """
        if generate_content_config_dict:
            self._normalize_response_schema(generate_content_config_dict, model)

        # Build the request in Google GenAI format that Vertex AI expects
        result: Final = {
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
