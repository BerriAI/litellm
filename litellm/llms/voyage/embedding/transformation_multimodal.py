"""
This module is used to transform the request and response for the Voyage multimodal
embeddings API (/v1/multimodalembeddings). Used for voyage-multimodal-3 and
voyage-multimodal-3.5.

Image (and video) embeddings: pass input in Voyage-native format so each item
can mix text and images/videos. Each element of input must be:
  {"content": [<content pieces>]}

Each content piece is one of:
  - {"type": "text", "text": "<string>"}
  - {"type": "image_url", "image_url": "<url>"}   # PNG, JPEG, WEBP, GIF
  - {"type": "image_base64", "image_base64": "<data URL>"}  # e.g. data:image/jpeg;base64,...
  - {"type": "video_url", "video_url": "<url>"}   # MP4
  - {"type": "video_base64", "video_base64": "<data URL>"}

Example with text + image URL:
  input=[{"content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": "https://example.com/photo.jpg"}
  ]}]
"""
from typing import List, Optional, Union, cast

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class VoyageError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, httpx.Headers] = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.voyageai.com/v1/multimodalembeddings"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class VoyageMultimodalEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.voyageai.com/reference/multimodal-embeddings-api
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
        if api_base:
            if not api_base.endswith("/multimodalembeddings"):
                api_base = f"{api_base}/multimodalembeddings"
            return api_base
        return "https://api.voyageai.com/v1/multimodalembeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return ["input_type", "truncation"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Voyage multimodal params.

        Reference: https://docs.voyageai.com/reference/multimodal-embeddings-api
        """
        if "input_type" in non_default_params:
            optional_params["input_type"] = non_default_params["input_type"]
        if "truncation" in non_default_params:
            optional_params["truncation"] = non_default_params["truncation"]
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
            api_key = (
                get_secret_str("VOYAGE_API_KEY")
                or get_secret_str("VOYAGE_AI_API_KEY")
                or get_secret_str("VOYAGE_AI_TOKEN")
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _normalize_input_to_voyage_multimodal(
        self, input: Union[AllEmbeddingInputValues, List[dict]]
    ) -> List[dict]:
        """
        Convert OpenAI-style input (str or list of str) to Voyage multimodal format.
        Voyage expects: inputs = [ {"content": [ {"type": "text", "text": "..."} ]}, ... ]
        If input is already a list of dicts with "content" key, return as-is.
        """
        if isinstance(input, str):
            return [{"content": [{"type": "text", "text": input}]}]
        if isinstance(input, list) and len(input) > 0:
            first = input[0]
            if isinstance(first, dict) and "content" in first:
                return cast(List[dict], input)  # already Voyage-native format
            if isinstance(first, str):
                return [
                    {"content": [{"type": "text", "text": item}]}
                    for item in input
                ]
            # list of token IDs not supported for multimodal; treat as invalid
        return [{"content": [{"type": "text", "text": ""}]}]

    def transform_embedding_request(
        self,
        model: str,
        input: Union[AllEmbeddingInputValues, List[dict]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        # Voyage API expects model name without provider prefix (e.g. voyage-multimodal-3.5)
        api_model = model.split("/", 1)[-1] if "/" in model else model
        inputs = self._normalize_input_to_voyage_multimodal(input)
        return {
            "inputs": inputs,
            "model": api_model,
            **optional_params,
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise VoyageError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage = Usage(
            prompt_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
            total_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
        )
        model_response.usage = usage
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return VoyageError(
            message=error_message, status_code=status_code, headers=headers
        )

    @staticmethod
    def is_multimodal_embedding(model: str) -> bool:
        return "multimodal" in model.lower()
