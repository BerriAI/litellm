import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import httpx

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.llms.openai import (
        HttpxBinaryResponseContent as _HttpxBinaryResponseContent,
    )

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
    HttpxBinaryResponseContent = _HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any
    HttpxBinaryResponseContent = Any


class TextToSpeechRequestData(TypedDict, total=False):
    """
    Structured return type for text-to-speech transformations.
    
    This ensures a consistent interface across all TTS providers.
    Providers should set ONE of: dict_body, ssml_body, or text_body.
    """
    dict_body: Dict[str, Any]  # JSON request body (e.g., OpenAI TTS)
    ssml_body: str  # SSML/XML string body (e.g., Azure AVA TTS)
    headers: Dict[str, str]  # Provider-specific headers to merge with base headers


class BaseTextToSpeechConfig(ABC):
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
    def get_supported_openai_params(self, model: str) -> list:
        """
        Get list of OpenAI TTS parameters supported by this provider
        """
        pass

    @abstractmethod
    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI TTS parameters to provider-specific parameters
        """
        pass

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and return headers
        """
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete url for the request
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    @abstractmethod
    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Transform request to provider-specific format.
        
        Returns:
            TextToSpeechRequestData: A structured dict containing:
                - body: The request body (JSON dict, XML string, or binary data)
                - headers: Provider-specific headers to merge with base headers
        """
        pass

    @abstractmethod
    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        """
        Transform provider response to standard format
        """
        pass

    def get_error_class(
        self, error_message: str, status_code: int, headers: Dict
    ) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

