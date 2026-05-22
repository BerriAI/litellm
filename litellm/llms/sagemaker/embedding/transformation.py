"""
Translate from OpenAI's `/v1/embeddings` to Sagemaker's `/invoke`

In the Huggingface TGI format.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response

import litellm
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import Usage, EmbeddingResponse
from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig

from ..common_utils import SagemakerError


class SagemakerCohereEmbeddingConfig(BaseEmbeddingConfig):
    """
    SageMaker embedding config for AWS Marketplace Cohere containers.

    These containers expose the native Cohere embed API, which expects
    ``{"texts": [...], "input_type": "..."}`` rather than the HuggingFace
    ``{"inputs": [...]}`` format that the default SageMaker config sends.

    Reference: https://docs.cohere.com/v2/reference/embed
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                optional_params["embedding_types"] = v if isinstance(v, list) else [v]
            elif k == "dimensions":
                optional_params["output_dimension"] = v
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
        Transform embedding request for Cohere models on SageMaker.

        Cohere containers expect the native Cohere API format:
        ``{"texts": [...], "input_type": "search_document", ...}``
        """
        texts: List[str] = (
            list(input)  # type: ignore[arg-type]
            if isinstance(input, (list, tuple))
            else [input]  # type: ignore[list-item]
        )
        request: dict = {
            "texts": texts,
            "input_type": litellm.COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
        }
        request.update(optional_params)
        return request

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
        Transform embedding response for Cohere models on SageMaker.

        The Cohere container returns:
        ``{"id": "...", "embeddings": [[...], ...], "texts": [...], ...}``
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise SagemakerError(
                message=f"Failed to parse response: {str(e)}",
                status_code=raw_response.status_code,
            )

        if not isinstance(response_data, dict) or "embeddings" not in response_data:
            raise SagemakerError(
                status_code=500,
                message=(
                    "Unexpected Cohere response format. "
                    f"Expected dict with 'embeddings' key, got: {type(response_data).__name__}"
                ),
            )

        embeddings = response_data["embeddings"]
        output_data = []

        if isinstance(embeddings, dict):
            # Cohere v2 containers return embeddings keyed by type when
            # ``embedding_types`` is set, e.g.:
            #   {"float": [[0.1, ...], ...], "int8": [[1, ...], ...]}
            # Follow the same flattening pattern used in the hosted
            # CohereEmbeddingConfig._transform_response so callers get the same
            # shape regardless of whether they hit the cloud API or a SageMaker
            # Marketplace container.
            for _embedding_type, embedding_list in embeddings.items():
                for idx, embedding in enumerate(embedding_list):
                    output_data.append(
                        {"object": "embedding", "index": idx, "embedding": embedding}
                    )
        else:
            # Flat list: the standard Cohere SageMaker response when
            # ``embedding_types`` is not set.
            # [[float, ...], [float, ...], ...]
            for idx, embedding in enumerate(embeddings):
                output_data.append(
                    {"object": "embedding", "index": idx, "embedding": embedding}
                )

        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model

        input_texts = request_data.get("texts", [])
        input_tokens = sum(len(text.split()) for text in input_texts)
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
        return {"Content-Type": "application/json"}


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
        elif "cohere" in model.lower():
            return SagemakerCohereEmbeddingConfig()
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
                status_code=raw_response.status_code,
            )

        # Handle both raw array format (TEI) and wrapped format (standard HF)
        if isinstance(response_data, list):
            # TEI and some HF models return raw embedding arrays directly
            embeddings = response_data
        elif isinstance(response_data, dict) and "embedding" in response_data:
            # Standard HF format with "embedding" key
            embeddings = response_data["embedding"]
        else:
            raise SagemakerError(
                status_code=500,
                message=f"Unexpected response format. Expected list or dict with 'embedding' key, got: {type(response_data).__name__}",
            )

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
