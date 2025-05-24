from typing import List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class CloudflareEmbeddingError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, httpx.Headers] = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST",
            url="https://api.cloudflare.com/client/v4/accounts/*/ai/run/*",
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class CloudflareEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://developers.cloudflare.com/workers-ai/models/bge-large-en-v1.5/
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            account_id = get_secret_str("CLOUDFLARE_ACCOUNT_ID")
            if account_id is None:
                raise ValueError(
                    "No CLOUDFLARE_ACCOUNT_ID provided. Set 'CLOUDFLARE_ACCOUNT_ID' in environment variables"
                )
            api_base = (
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
            )
        return api_base + model

    def get_supported_openai_params(self, model: str) -> list:
        # BGE models support basic embedding parameters
        return [
            "encoding_format",  # float or base64
            "dimensions",  # output dimensions (model dependent)
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI embedding params to Cloudflare BGE params

        Reference: https://developers.cloudflare.com/workers-ai/models/bge-large-en-v1.5/
        """
        # Cloudflare BGE models have their own parameter structure
        # For now, we'll focus on the basic text input transformation
        return optional_params

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
        if api_key is None:
            api_key = get_secret_str("CLOUDFLARE_API_KEY")

        if api_key is None:
            raise ValueError(
                "Missing Cloudflare API Key - A call is being made to Cloudflare but no key is set either in the environment variables or via params"
            )

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the embedding request to Cloudflare BGE format

        Cloudflare BGE models expect:
        {
            "text": ["text1", "text2", ...]
        }
        """
        # Ensure input is a list of strings
        text_input: list[str]
        if isinstance(input, str):
            text_input = [input]
        elif isinstance(input, list):
            # Handle different list types
            if all(isinstance(item, str) for item in input):
                text_input = input  # type: ignore[assignment]
            elif all(isinstance(item, int) for item in input):
                # Convert list of integers to list of strings
                text_input = [str(item) for item in input]
            elif all(isinstance(item, list) for item in input):
                # Convert list of lists to list of strings (join inner lists)
                text_input = [str(item) for item in input]
            else:
                # Mixed types or unsupported types
                text_input = [str(item) for item in input]
        else:
            raise ValueError(
                f"Invalid input type for Cloudflare embedding: {type(input)}"
            )

        data = {"text": text_input}

        # Add any model-specific parameters
        data.update(optional_params)

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
        """
        Transform Cloudflare BGE response to OpenAI embedding format

        Cloudflare response format:
        {
            "result": {
                "shape": [num_texts, dimensions],
                "data": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            },
            "success": true,
            "errors": [],
            "messages": []
        }
        """
        response_json = raw_response.json()

        if not response_json.get("success", True):
            errors = response_json.get("errors", [])
            error_message = "; ".join([str(err) for err in errors])
            raise CloudflareEmbeddingError(
                status_code=400, message=f"Cloudflare API error: {error_message}"
            )

        result = response_json.get("result", {})

        # Extract embeddings from Cloudflare format
        embeddings_data = result.get("data", [])

        # Transform to OpenAI format
        embeddings = []
        for i, embedding in enumerate(embeddings_data):
            embeddings.append(
                {"object": "embedding", "index": i, "embedding": embedding}
            )

        # Set usage if available
        usage = None
        if "usage" in result:
            usage_data = result["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=0,  # Embeddings don't have completion tokens
                total_tokens=usage_data.get(
                    "total_tokens", usage_data.get("prompt_tokens", 0)
                ),
            )

        # Update model response
        model_response.object = "list"
        model_response.data = embeddings  # type: ignore
        model_response.model = f"cloudflare/{model}"
        model_response.usage = usage  # type: ignore

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return CloudflareEmbeddingError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
