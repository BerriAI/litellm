"""
Dynamic embedding config for JSON-configured OpenAI-compatible providers
"""
from typing import List, Optional

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse


class JSONProviderEmbeddingConfig(BaseEmbeddingConfig):
    """
    Embedding config for JSON-configured OpenAI-compatible providers.
    Works with any provider defined in providers.json.
    """

    def __init__(self, provider_config):
        self.provider_config = provider_config
        self.base_url = provider_config.base_url
        self.api_key_env = provider_config.api_key_env
        self.api_base_env = provider_config.api_base_env

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get the complete URL for the embedding endpoint."""
        if api_base is None:
            api_base = self.base_url
        if api_base is None:
            raise ValueError(f"No API base URL configured for provider")
        api_base = api_base.rstrip("/")
        complete_url = f"{api_base}/embeddings"
        return complete_url

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
        """Validate and set up authentication headers."""
        if api_key is None:
            api_key = get_secret_str(self.api_key_env)

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        return {**default_headers, **headers}

    def get_supported_openai_params(self, model: str) -> List[str]:
        """Get supported OpenAI parameters for embedding requests."""
        return [
            "dimensions",
            "encoding_format",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map OpenAI parameters to provider format."""
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """Transform the embedding request into provider format."""
        return {"input": input, "model": model, **optional_params}

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
        """Transform the embedding response from provider format."""
        import json
        from litellm.types.utils import Embedding, Usage

        try:
            response_json = raw_response.json()
        except Exception:
            raise ValueError(f"Invalid JSON response: {raw_response.text}")

        # Transform data array - OpenAI format
        if "data" in response_json and len(response_json["data"]) > 0:
            embedding_objects = []
            for item in response_json["data"]:
                if isinstance(item, dict):
                    embedding_obj = Embedding(
                        embedding=item.get("embedding", []),
                        index=item.get("index", 0),
                        object=item.get("object", "embedding"),
                    )
                    embedding_objects.append(embedding_obj)
            model_response.data = embedding_objects

        # Transform usage
        if "usage" in response_json:
            usage_data = response_json["usage"]
            if isinstance(usage_data, dict):
                model_response.usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

        # Set model
        if "model" in response_json:
            model_response.model = response_json["model"]
        else:
            model_response.model = model

        # Set object type
        model_response.object = response_json.get("object", "list")

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict
    ):
        """Get the appropriate error class for JSON provider exceptions."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException
        from litellm.exceptions import (
            APIError,
            APIConnectionError,
            RateLimitError,
            Timeout,
        )

        if status_code == 429:
            return RateLimitError(message=error_message, status_code=status_code, headers=headers)
        elif status_code == 408 or status_code == 504:
            return Timeout(message=error_message, status_code=status_code, headers=headers)
        elif status_code >= 500:
            # APIConnectionError doesn't take status_code, just message
            return APIConnectionError(message=error_message)
        else:
            return APIError(message=error_message, status_code=status_code, headers=headers)


def create_json_embedding_config(provider_config):
    """Create an embedding config for a JSON-configured provider."""
    return JSONProviderEmbeddingConfig(provider_config)

