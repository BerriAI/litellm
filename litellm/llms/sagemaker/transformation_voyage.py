"""
Voyage AI transformation for SageMaker endpoints

This module handles the transformation of Voyage AI embedding requests
and responses when deployed on AWS SageMaker endpoints.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response

from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import EmbeddingResponse, Usage



class VoyageSagemakerError(BaseLLMException):
    """Custom error class for Voyage SageMaker operations"""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, Headers] = {},
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class VoyageSagemakerEmbeddingConfig(BaseEmbeddingConfig):
    """
    Voyage AI embedding configuration for SageMaker endpoints
    
    This class handles the transformation of Voyage AI embedding requests
    and responses when deployed on AWS SageMaker endpoints.
    
    Reference: https://docs.voyageai.com/reference/embeddings-api
    """
    
    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Get supported OpenAI parameters for Voyage models on SageMaker
        
        Args:
            model: The model name
            
        Returns:
            List of supported parameter names
        """
        return [
            "encoding_format",
            "dimensions",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Voyage SageMaker format
        
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
                # Voyage supports: float, base64
                if value in ["float", "base64"]:
                    optional_params["encoding_format"] = value
                else:
                    if not drop_params:
                        raise ValueError(
                            f"Unsupported encoding_format: {value}. Voyage supports: float, base64"
                        )
            elif param == "dimensions":
                # Map OpenAI dimensions to Voyage output_dimension
                optional_params["output_dimension"] = value
            elif not drop_params:
                raise ValueError(f"Unsupported parameter for Voyage SageMaker: {param}")
                
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: "AllEmbeddingInputValues",
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request for Voyage models on SageMaker
        
        Args:
            model: The model name
            input: List of input texts to embed
            optional_params: Optional parameters
            headers: Request headers
            
        Returns:
            Transformed request data
        """
        # Voyage models on SageMaker expect "input" field (singular)
        request_data = {
            "input": input,
            **optional_params
        }
        
        return request_data

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: EmbeddingResponse,
        logging_obj: Any,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        """
        Transform embedding response from Voyage SageMaker format to OpenAI format
        
        Args:
            model: The model name
            raw_response: Raw HTTP response
            model_response: EmbeddingResponse object to populate
            logging_obj: Logging object
            api_key: API key (unused)
            request_data: Original request data
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            Transformed EmbeddingResponse
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise VoyageSagemakerError(
                message=f"Failed to parse response: {str(e)}", 
                status_code=raw_response.status_code
            )

        # Voyage response format:
        # {
        #   "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
        #   "object": "list",
        #   "model": "voyage-code-02",
        #   "usage": {"total_tokens": 10}
        # }
        
        if "data" not in response_data:
            raise VoyageSagemakerError(
                status_code=500, 
                message="Voyage response missing 'data' field"
            )
        
        embeddings = response_data["data"]
        if not isinstance(embeddings, list):
            raise VoyageSagemakerError(
                status_code=422,
                message=f"Voyage response data not in expected format - {embeddings}",
            )

        output_data = []
        for idx, embedding_item in enumerate(embeddings):
            if "embedding" not in embedding_item:
                raise VoyageSagemakerError(
                    status_code=500, 
                    message=f"Voyage embedding item {idx} missing 'embedding' field"
                )
            
            output_data.append({
                "object": "embedding",
                "index": idx,
                "embedding": embedding_item["embedding"]
            })

        model_response.object = response_data.get("object", "list")
        model_response.data = output_data
        model_response.model = response_data.get("model", model)

        # Calculate usage
        total_tokens = response_data.get("usage", {}).get("total_tokens", 0)
        if total_tokens == 0:
            # Fallback: calculate tokens from input
            input_tokens = 0
            for text in request_data.get("input", []):
                input_tokens += len(text.split())  # Simple word count fallback
            total_tokens = input_tokens

        model_response.usage = Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        )

        return model_response

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment for Voyage SageMaker embeddings
        
        Args:
            headers: Request headers
            model: Model name
            messages: Messages (unused for embeddings)
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            api_key: API key (unused)
            api_base: API base (unused)
            
        Returns:
            Validated headers
        """
        return {
            "Content-Type": "application/json",
            **headers
        }

    def get_error_class(
        self, 
        error_message: str, 
        status_code: int, 
        headers: Union[dict, Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate error class for this configuration
        
        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers
            
        Returns:
            Appropriate exception instance
        """
        return VoyageSagemakerError(
            message=error_message, 
            status_code=status_code, 
            headers=headers
        )
