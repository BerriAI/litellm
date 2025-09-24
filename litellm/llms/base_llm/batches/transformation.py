import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx
from httpx import Headers

from litellm.types.llms.openai import (
    AllMessageValues,
    CreateBatchRequest,
)
from litellm.types.utils import LiteLLMBatch, LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class BaseBatchesConfig(ABC):
    """
    Abstract base class for batch processing configurations across different LLM providers.
    
    This class defines the interface that all provider-specific batch configurations
    must implement to work with LiteLLM's unified batch processing system.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        """Return the LLM provider type for this configuration."""
        pass

    @classmethod
    def get_config(cls):
        """Get configuration dictionary for this class."""
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
        """
        Validate and prepare environment-specific headers and parameters.
        
        Args:
            headers: HTTP headers dictionary
            model: Model name
            messages: List of messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            api_key: API key
            api_base: API base URL
            
        Returns:
            Updated headers dictionary
        """
        pass

    @abstractmethod
    def get_complete_batch_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateBatchRequest,
    ) -> str:
        """
        Get the complete URL for batch creation request.
        
        Args:
            api_base: Base API URL
            api_key: API key
            model: Model name
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            data: Batch creation request data
            
        Returns:
            Complete URL for the batch request
        """
        pass

    @abstractmethod
    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Transform the batch creation request to provider-specific format.
        
        Args:
            model: Model name
            create_batch_data: Batch creation request data
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            Transformed request data
        """
        pass

    @abstractmethod
    def transform_create_batch_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """
        Transform provider-specific batch response to LiteLLM format.
        
        Args:
            model: Model name
            raw_response: Raw HTTP response
            logging_obj: Logging object
            litellm_params: LiteLLM parameters
            
        Returns:
            LiteLLM batch object
        """
        pass

    @abstractmethod
    def transform_retrieve_batch_request(
        self,
        batch_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Transform the batch retrieval request to provider-specific format.
        
        Args:
            batch_id: Batch ID to retrieve
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            Transformed request data
        """
        pass

    @abstractmethod
    def transform_retrieve_batch_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """
        Transform provider-specific batch retrieval response to LiteLLM format.
        
        Args:
            model: Model name
            raw_response: Raw HTTP response
            logging_obj: Logging object
            litellm_params: LiteLLM parameters
            
        Returns:
            LiteLLM batch object
        """
        pass

    @abstractmethod
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> "BaseLLMException":
        """
        Get the appropriate error class for this provider.
        
        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers
            
        Returns:
            Provider-specific exception class
        """
        pass
