"""
Volcengine Embedding Handler
Handles embedding requests to Volcengine's embedding API
"""

from typing import Dict, List, Optional, Union, Any

import httpx
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.types.utils import EmbeddingResponse
import litellm

from .transformation import VolcEngineEmbeddingConfig
from ..common_utils import VolcEngineError


class VolcEngineEmbeddingHandler:
    """Handler for Volcengine embedding API calls"""

    def __init__(self):
        self.config = VolcEngineEmbeddingConfig()

    def _convert_to_litellm_response(self, transformed_response: Dict, model: str, input: Union[str, List[str]]) -> EmbeddingResponse:
        """Convert transformed response to LiteLLM EmbeddingResponse"""
        model_response = EmbeddingResponse()
        model_response.object = transformed_response.get("object", "list")
        model_response.data = transformed_response.get("data", [])
        model_response.model = transformed_response.get("model", model)
        
        # Set usage information
        usage_data = transformed_response.get("usage", {})
        if usage_data:
            model_response.usage = litellm.Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=0,
                total_tokens=usage_data.get("total_tokens", usage_data.get("prompt_tokens", 0)),
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )
        
        return model_response

    def embedding(
        self,
        model: str,
        input: Union[str, List[str]],
        api_key: str,
        api_base: Optional[str] = None,
        encoding_format: Optional[str] = "float",
        user: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = None,
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Synchronous embedding call to Volcengine API.

        Args:
            model: Volcengine model ID (e.g., "doubao-embedding-text-240715")
            input: Text or list of texts to embed
            api_key: Volcengine API key
            api_base: Optional custom API base URL
            encoding_format: Response format (float, base64, null)
            user: Optional user identifier
            timeout: Request timeout
            extra_headers: Optional additional headers
            litellm_logging_obj: Optional logging object
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse object
        """
        # Transform request to Volcengine format
        request_data = self.config.transform_request(
            model=model,
            input=input,
            api_key=api_key,
            api_base=api_base,
            encoding_format=encoding_format,
            user=user,
            extra_headers=extra_headers,
            **kwargs,
        )

        # Make HTTP request
        try:
            client = HTTPHandler(timeout=timeout)
            response = client.post(
                url=request_data["url"],
                headers=request_data["headers"],
                json=request_data["data"],
            )
        except Exception as e:
            raise VolcEngineError(
                status_code=500,
                message=f"Network error during embedding request: {str(e)}",
            )

        # Handle HTTP errors
        if response.status_code != 200:
            error_message = f"Volcengine embedding request failed with status {response.status_code}"
            try:
                error_details = response.json()
                if "error" in error_details:
                    error_message += f": {error_details['error']}"
                elif "message" in error_details:
                    error_message += f": {error_details['message']}"
            except Exception:
                error_message += f": {response.text}"

            raise VolcEngineError(
                status_code=response.status_code,
                message=error_message,
                headers=response.headers,
            )

        # Transform response to OpenAI format
        transformed_response = self.config.transform_response(
            response=response, model=model, input=input, encoding=encoding_format
        )

        # Convert to LiteLLM EmbeddingResponse
        return self._convert_to_litellm_response(transformed_response, model, input)

    async def async_embedding(
        self,
        model: str,
        input: Union[str, List[str]],
        api_key: str,
        api_base: Optional[str] = None,
        encoding_format: Optional[str] = "float",
        user: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = None,
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Asynchronous embedding call to Volcengine API.

        Args:
            model: Volcengine model ID (e.g., "doubao-embedding-text-240715")
            input: Text or list of texts to embed
            api_key: Volcengine API key
            api_base: Optional custom API base URL
            encoding_format: Response format (float, base64, null)
            user: Optional user identifier
            timeout: Request timeout
            extra_headers: Optional additional headers
            litellm_logging_obj: Optional logging object
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse object
        """
        # Transform request to Volcengine format
        request_data = self.config.transform_request(
            model=model,
            input=input,
            api_key=api_key,
            api_base=api_base,
            encoding_format=encoding_format,
            user=user,
            extra_headers=extra_headers,
            **kwargs,
        )

        # Make async HTTP request
        try:
            client = AsyncHTTPHandler(timeout=timeout)
            response = await client.post(
                url=request_data["url"],
                headers=request_data["headers"],
                json=request_data["data"],
            )
        except Exception as e:
            raise VolcEngineError(
                status_code=500,
                message=f"Network error during embedding request: {str(e)}",
            )

        # Handle HTTP errors
        if response.status_code != 200:
            error_message = f"Volcengine embedding request failed with status {response.status_code}"
            try:
                error_details = response.json()
                if "error" in error_details:
                    error_message += f": {error_details['error']}"
                elif "message" in error_details:
                    error_message += f": {error_details['message']}"
            except Exception:
                error_message += f": {response.text}"

            raise VolcEngineError(
                status_code=response.status_code,
                message=error_message,
                headers=response.headers,
            )

        # Transform response to OpenAI format
        transformed_response = self.config.transform_response(
            response=response, model=model, input=input, encoding=encoding_format
        )

        # Convert to LiteLLM EmbeddingResponse
        return self._convert_to_litellm_response(transformed_response, model, input)
