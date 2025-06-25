import types
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import httpx

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
        pass

    @abstractmethod
    def map_generate_content_optional_params(
        self,
        generate_content_optional_params: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        """
        Map Google GenAI parameters to provider-specific format.

        Args:
            generate_content_optional_params: Optional parameters for generate content
            model: The model name

        Returns:
            Mapped parameters for the provider
        """
        pass

    @abstractmethod
    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate the environment and return headers for the request.

        Args:
            headers: Existing headers
            model: The model name
            litellm_params: LiteLLM parameters

        Returns:
            Updated headers
        """
        return headers

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
        contents: Union[str, List[Dict[str, Any]]],
        generate_content_request_params: Dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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
