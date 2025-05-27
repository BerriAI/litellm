"""
Translate from OpenAI's `/v1/embeddings` to Morph's `/v1/embeddings`
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import EmbeddingResponse


class MorphError(BaseLLMException):
    """
    Exception raised for Morph API errors.
    """
    pass


class MorphEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.morphllm.com/api-reference/endpoint/embeddings
    
    Morph provides an OpenAI-compatible embeddings API.
    """
    
    def __init__(self) -> None:
        pass
        
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Morph is OpenAI compatible, set to custom_openai and use Morph's endpoint
        api_base = (
            api_base
            or get_secret_str("MORPH_API_BASE")
            or "https://api.morphllm.com/v1"
        )
        dynamic_api_key = api_key or get_secret_str("MORPH_API_KEY")
        return api_base, dynamic_api_key
    
    def get_supported_openai_params(self, model: str) -> List[str]:
        """Get the list of parameters supported by Morph embeddings"""
        return [
            "input",
            "model",
            "dimensions",
            "encoding_format",
            "user"
        ]
    
    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool = False,
    ) -> Dict[str, Any]:
        """Map OpenAI parameters to Morph parameters"""
        supported_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_params:
                optional_params[param] = value
        return optional_params
    
    def transform_embedding_request(
        self,
        model: str,
        input: Union[str, List[str]],
        optional_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transform embedding request for Morph API"""
        request_data = {
            "model": model.replace("morph/", ""),
            "input": input,
        }
        
        # Add optional parameters
        for param in self.get_supported_openai_params(model):
            if param in optional_params and param not in ["model", "input"]:
                request_data[param] = optional_params[param]
                
        return request_data
    
    def transform_embedding_response(
        self,
        model: str,
        response: httpx.Response,
        embedding_response: EmbeddingResponse,
    ) -> EmbeddingResponse:
        """Transform embedding response from Morph API"""
        try:
            response_json = response.json()
            # Morph follows OpenAI format so we don't need any special transformations
            return EmbeddingResponse(**response_json)
        except Exception:
            raise MorphError(
                message=response.text,
                status_code=response.status_code
            )
        
    def validate_environment(
        self,
        headers: Dict[str, Any],
        model: str,
        messages: Optional[List[AllMessageValues]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate that the necessary API key is available."""
        if optional_params is None:
            optional_params = {}
            
        if api_key is None:
            api_key = get_secret_str("MORPH_API_KEY")

        if api_key is None:
            raise ValueError("Morph API key is required. Please set 'MORPH_API_KEY' environment variable.")

        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        
        return headers
    
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict[str, Any], httpx.Headers]
    ) -> BaseLLMException:
        """Return the appropriate error class for Morph API errors"""
        return MorphError(message=error_message, status_code=status_code)
