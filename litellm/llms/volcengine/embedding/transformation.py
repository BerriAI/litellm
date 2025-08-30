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

    def transform_request(
        self,
        model: str,
        input: Union[str, List[str]],
        api_key: str,
        api_base: Optional[str] = None,
        encoding_format: Optional[str] = "float",
        user: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transform OpenAI embedding request to Volcengine format.

        Args:
            model: Model ID (e.g., "doubao-embedding-text-240715")
            input: Text or list of texts to embed
            api_key: Volcengine API key
            api_base: Optional custom API base URL
            encoding_format: Response format (float, base64, null)
            user: Optional user identifier
            extra_headers: Optional additional headers
            **kwargs: Additional parameters

        Returns:
            Dict containing url, headers, and data for the request
        """
        # Get base URL
        base_url = get_volcengine_base_url(api_base)
        # Avoid duplicate /api/v3 if base_url already contains it
        if base_url.endswith("/api/v3"):
            url = f"{base_url}/embeddings"
        else:
            url = f"{base_url}/api/v3/embeddings"

        # Get headers
        headers = get_volcengine_headers(api_key, extra_headers)

        # Prepare request data
        data = {
            "model": model,
            "input": input if isinstance(input, list) else [input],
        }

        # Add optional parameters
        if encoding_format is not None:
            data["encoding_format"] = encoding_format

        return {
            "url": url,
            "headers": headers,
            "data": data,
        }

    def transform_response(
        self,
        response: httpx.Response,
        model: str,
        input: Union[str, List[str]],
        encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transform Volcengine embedding response to OpenAI format.

        Args:
            response: The HTTP response from Volcengine
            model: The model used
            input: The input that was embedded
            encoding: The encoding format requested

        Returns:
            OpenAI-compatible embedding response
        """
        try:
            response_json = response.json()
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

        return transformed_response

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """Transform embedding request to Volcengine format"""
        # Use existing transform_request method
        return self.transform_request(
            model=model,
            input=input,
            api_key="",  # api_key will be in headers
            **optional_params,
        )

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
        # Use existing transform_response method
        transformed_response = self.transform_response(
            response=raw_response,
            model=model,
            input=request_data.get("input", []),
        )
        
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
        volcengine_headers = get_volcengine_headers(api_key)
        return {**headers, **volcengine_headers}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """Get error class for Volcengine errors"""
        from ..common_utils import VolcEngineError
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
