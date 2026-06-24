"""
Hosted VLLM Embedding API Configuration.

This module provides the configuration for hosted VLLM's Embedding API.
VLLM is OpenAI-compatible and supports embeddings via the /v1/embeddings endpoint.

Docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import BadRequestError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class _ImageUrlField(BaseModel):
    url: str | None = None


class _ContentPart(BaseModel):
    image_url: str | _ImageUrlField | None = None


class _MessageWithImages(BaseModel):
    content: str | list[_ContentPart] | None = None


_MESSAGES_ADAPTER: TypeAdapter[list[_MessageWithImages]] = TypeAdapter(
    list[_MessageWithImages]
)


def _reject_non_data_image_urls(messages: object, model: str) -> None:
    """Reject any image URL vLLM would fetch server-side, to prevent SSRF.

    Only base64 `data:` URLs are allowed; vLLM never makes an outbound request for
    those. A remote `image_url` (http(s) or any other scheme) would let an
    authenticated caller make the vLLM host fetch arbitrary internal endpoints, so
    it is rejected before forwarding.
    """
    try:
        parsed = _MESSAGES_ADAPTER.validate_python(messages)
    except ValidationError:
        return
    for message in parsed:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            image_url = part.image_url
            url = image_url.url if isinstance(image_url, _ImageUrlField) else image_url
            if url is None or url.startswith("data:"):
                continue
            raise BadRequestError(
                message=(
                    "hosted_vllm embeddings: only base64 `data:` image URLs are "
                    "supported in `messages`. A remote image URL was rejected to "
                    "prevent server-side request forgery from the vLLM host."
                ),
                model=model,
                llm_provider="hosted_vllm",
            )


class HostedVLLMEmbeddingError(BaseLLMException):
    """Exception class for Hosted VLLM Embedding errors."""

    pass


class HostedVLLMEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for Hosted VLLM's Embedding API.

    Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

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
        Validate environment and set up headers for Hosted VLLM API.
        """
        if api_key is None:
            api_key = get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"

        default_headers = {
            "Content-Type": "application/json",
        }

        # Only add Authorization header if api_key is not "fake-api-key"
        if api_key and api_key != "fake-api-key":
            default_headers["Authorization"] = f"Bearer {api_key}"

        # Merge with existing headers (user's headers take priority)
        return {**default_headers, **headers}

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
        Get the complete URL for Hosted VLLM Embedding API endpoint.
        """
        if api_base is None:
            api_base = get_secret_str("HOSTED_VLLM_API_BASE")
            if api_base is None:
                raise ValueError("api_base is required for hosted_vllm embeddings")

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Ensure the URL ends with /embeddings
        if not api_base.endswith("/embeddings"):
            api_base = f"{api_base}/embeddings"

        return api_base

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request to Hosted VLLM format (OpenAI-compatible).
        """
        # Strip 'hosted_vllm/' prefix if present
        if model.startswith("hosted_vllm/"):
            model = model.replace("hosted_vllm/", "", 1)

        # vLLM's embeddings endpoint accepts a chat-style `messages` body for multimodal
        # (image) embeddings; forward it instead of `input` when present
        if "messages" in optional_params:
            messages = optional_params["messages"]
            _reject_non_data_image_urls(messages, model)
            if input:
                verbose_logger.debug(
                    "hosted_vllm embeddings: both `messages` and `input` provided; "
                    "forwarding `messages` and ignoring `input`"
                )
            remaining_params = {
                k: v for k, v in optional_params.items() if k != "messages"
            }
            return {
                "model": model,
                "messages": messages,
                **remaining_params,
            }

        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        return {
            "model": model,
            "input": input,
            **optional_params,
        }

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
        Transform embedding response from Hosted VLLM format (OpenAI-compatible).
        """
        logging_obj.post_call(original_response=raw_response.text)

        # VLLM returns standard OpenAI-compatible embedding response
        response_json = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get list of supported OpenAI parameters for Hosted VLLM embeddings.
        """
        return [
            "timeout",
            "dimensions",
            "encoding_format",
            "user",
            "messages",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Hosted VLLM format.
        """
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the error class for Hosted VLLM errors.
        """
        return HostedVLLMEmbeddingError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
