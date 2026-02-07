"""
Translate from OpenAI's `/v1/embeddings` to Sagemaker's `/invoke`

In the Huggingface TGI format. 
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response

from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import Usage, EmbeddingResponse
from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig

from ..common_utils import SagemakerError


class SagemakerEmbeddingConfig(BaseEmbeddingConfig):
    """
    SageMaker embedding configuration factory for supporting embedding parameters
    """
    
    def __init__(self) -> None:
        pass

    @classmethod
    def get_model_config(cls, model: str) -> "BaseEmbeddingConfig":
        """
        Factory method to get the appropriate embedding config based on model type
        
        Args:
            model: The model name
            
        Returns:
            Appropriate embedding config instance
        """
        if "voyage" in model.lower():
            return VoyageEmbeddingConfig()
        else:
            return cls()

    def get_supported_openai_params(self, model: str) -> List[str]:
        # Check if this is an embedding model
        if "voyage" in model.lower():
            return VoyageEmbeddingConfig().get_supported_openai_params(model)
        else:
            return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
    
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SagemakerError(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_embedding_request(
        self,
        model: str,
        input: "AllEmbeddingInputValues",
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request for Hugging Face models on SageMaker
        """
        # HF models expect "inputs" field (plural)
        return {"inputs": input, **optional_params}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: "EmbeddingResponse",
        logging_obj: Any,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> "EmbeddingResponse":
        """
        Transform embedding response for Hugging Face models on SageMaker
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise SagemakerError(
                message=f"Failed to parse response: {str(e)}", 
                status_code=raw_response.status_code
            )

        if "embedding" not in response_data:
            raise SagemakerError(
                status_code=500, message="HF response missing 'embedding' field"
            )
        embeddings = response_data["embedding"]

        if not isinstance(embeddings, list):
            raise SagemakerError(
                status_code=422,
                message=f"HF response not in expected format - {embeddings}",
            )

        output_data = []
        for idx, embedding in enumerate(embeddings):
            output_data.append(
                {"object": "embedding", "index": idx, "embedding": embedding}
            )

        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model

        # Calculate usage from request data
        input_texts = request_data.get("inputs", [])
        input_tokens = 0
        for text in input_texts:
            input_tokens += len(text.split())  # Simple word count fallback

        model_response.usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
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
        Validate environment for SageMaker embeddings
        """
        return {"Content-Type": "application/json"}
