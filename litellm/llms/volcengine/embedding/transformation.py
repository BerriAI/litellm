"""
Volcengine Embedding Transformation
Transforms OpenAI embedding requests to Volcengine format
"""

from typing import List, Optional, Union, Dict, Any
import httpx
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from ..common_utils import get_volcengine_base_url, get_volcengine_headers


class VolcEngineEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for Volcengine embedding models.
    Reference: https://ark.cn-beijing.volces.com/api/v3/embeddings
    """

    def __init__(
        self,
        encoding_format: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Get the list of OpenAI parameters supported by Volcengine embedding models.

        Args:
            model: The model name

        Returns:
            List of supported parameter names
        """
        return [
            "encoding_format",
            "user",
            "extra_headers",
        ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for volcengine embedding API calls.
        
        Args:
            api_base: Optional custom API base URL
            api_key: API key (not used for URL construction)
            model: Model name (not used for URL construction)
            optional_params: Optional parameters (not used for URL construction)
            litellm_params: LiteLLM parameters (not used for URL construction)
            stream: Stream parameter (not used for URL construction)
            
        Returns:
            Complete URL for the embedding API endpoint
        """
        base_url = get_volcengine_base_url(api_base)
        # Construct the complete URL with /embeddings endpoint
        if base_url.endswith("/api/v3"):
            return f"{base_url}/embeddings"
        else:
            return f"{base_url}/api/v3/embeddings"

    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        """
        Map OpenAI embedding parameters to Volcengine format.

        Args:
            non_default_params: Parameters that are not default values
            optional_params: Optional parameters dict to update
            model: The model name
            drop_params: Whether to drop unsupported parameters

        Returns:
            Updated optional_params dict
        """
        for param, value in non_default_params.items():
            if param == "encoding_format":
                # Volcengine supports: float, base64, null
                if value in ["float", "base64", None]:
                    optional_params["encoding_format"] = value
                else:
                    if not drop_params:
                        raise ValueError(
                            f"Unsupported encoding_format: {value}. Volcengine supports: float, base64, null"
                        )
            elif param == "user":
                # Keep user parameter as-is
                optional_params["user"] = value
            elif param in self.get_supported_openai_params(model):
                optional_params[param] = value
            elif not drop_params:
                raise ValueError(f"Unsupported parameter for Volcengine: {param}")

        return optional_params



    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """Transform embedding request to Volcengine format"""
        # Prepare request data (only the JSON body, not the full request)
        data = {
            "model": model,
            "input": input if isinstance(input, list) else [input],
        }

        # Add optional parameters from optional_params
        if "encoding_format" in optional_params:
            encoding_format = optional_params["encoding_format"]
            if encoding_format is not None:
                data["encoding_format"] = encoding_format

        if "user" in optional_params:
            user = optional_params["user"]
            if user is not None:
                data["user"] = user

        return data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        """Transform Volcengine response to EmbeddingResponse"""
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse Volcengine response as JSON: {str(e)}")

        # Volcengine response format matches OpenAI format closely
        # Just need to ensure all required fields are present
        transformed_response = {
            "object": "list",
            "data": response_json.get("data", []),
            "model": response_json.get("model", model),
            "usage": response_json.get("usage", {}),
        }

        # Add id if present
        if "id" in response_json:
            transformed_response["id"] = response_json["id"]
        
        # Create EmbeddingResponse from transformed data
        return EmbeddingResponse(**transformed_response)

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
        """Validate environment and return headers"""
        # Get Volcengine headers
        if api_key is None:
            raise ValueError("api_key is required for Volcengine authentication")
        volcengine_headers = get_volcengine_headers(api_key)
        return {**headers, **volcengine_headers}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """Get error class for Volcengine errors"""
        from ..common_utils import VolcEngineError
        # Convert dict to httpx.Headers if needed
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
