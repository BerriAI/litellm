import types
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.types.google_genai.main import (
    GenerateContentConfigDict,
    GenerateContentContentListUnionDict,
    GenerateContentResponse,
)
from litellm.types.router import GenericLiteLLMParams


class BaseGoogleGenAIGenerateContentConfig(ABC):
    """Base configuration class for Google GenAI generate_content functionality"""

    def __init__(self):
        pass

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @abstractmethod
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


    @abstractmethod
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

    @abstractmethod
    def validate_environment(
        self, 
        api_key: Optional[str],
        headers: Optional[dict],
        model: str,
        litellm_params: Optional[Union[GenericLiteLLMParams, dict]]
    ) -> dict:
        """
        Validate the environment and return headers for the request.

        Args:
            api_key: API key
            headers: Existing headers
            model: The model name
            litellm_params: LiteLLM parameters

        Returns:
            Updated headers
        """
        raise NotImplementedError("validate_environment is not implemented")

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for the request.

        Args:
            api_base: Base API URL
            model: The model name
            litellm_params: LiteLLM parameters

        Returns:
            Complete URL for the API request
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    @abstractmethod
    def transform_generate_content_request(
        self,
        model: str,
        contents: GenerateContentContentListUnionDict,
        generate_content_config_dict: GenerateContentConfigDict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """
        Transform the request parameters for the generate content API.

        Args:
            model: The model name
            contents: Input contents
            generate_content_request_params: Request parameters
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Transformed request data
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
    ) -> Dict[str, Any]:
        """
        Transform a parsed streaming response chunk.

        Args:
            model: The model name
            parsed_chunk: Parsed chunk data

        Returns:
            Transformed chunk data
        """
        pass

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> Exception:
        """
        Get the appropriate exception class for the error.

        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers

        Returns:
            Exception instance
        """
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
