"""
Transformation logic for the Voyage multimodal embeddings API.

Used for all multimodal embedding models in Voyage (voyage-multimodal-3, voyage-multimodal-3.5).

Reference: https://docs.voyageai.com/reference/multimodal-embeddings-api
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class VoyageMultimodalEmbeddingError(BaseLLMException):
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
            url="https://api.voyageai.com/v1/multimodalembeddings",
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
        return ["encoding_format", "dimensions", "input_type"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        if "encoding_format" in non_default_params:
            optional_params["output_encoding"] = non_default_params["encoding_format"]
        if "dimensions" in non_default_params:
            optional_params["output_dimension"] = non_default_params["dimensions"]
        if "input_type" in non_default_params:
            optional_params["input_type"] = non_default_params["input_type"]
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
        }

    @staticmethod
    def _convert_input_to_multimodal(
        input: Any,
    ) -> List[Dict[str, Any]]:
        """Convert various input formats to Voyage multimodal input format.

        Supports:
        - str: single text -> [{"content": [{"type": "text", "text": "..."}]}]
        - List[str]: multiple texts -> [{"content": [{"type": "text", "text": "..."}]}, ...]
        - List[dict]: already structured content blocks (OpenAI-style or Voyage-native)
        """
        if isinstance(input, str):
            return [{"content": [{"type": "text", "text": input}]}]

        if isinstance(input, list) and len(input) > 0:
            if isinstance(input[0], str):
                return [{"content": [{"type": "text", "text": text}]} for text in input]

            if isinstance(input[0], dict):
                inputs = []
                for item in input:
                    if "content" in item:
                        content = item["content"]
                        converted = []
                        for block in content:
                            converted.append(_convert_content_block(block))
                        inputs.append({"content": converted})
                    else:
                        inputs.append({"content": [_convert_content_block(item)]})
                return inputs

        return [{"content": [{"type": "text", "text": str(input)}]}]

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        inputs = self._convert_input_to_multimodal(input)
        return {
            "inputs": inputs,
            "model": model,
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
            raise VoyageMultimodalEmbeddingError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage_data = raw_response_json.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("total_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        model_response.usage = usage
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return VoyageMultimodalEmbeddingError(
            message=error_message, status_code=status_code, headers=headers
        )

    @staticmethod
    def is_multimodal_embeddings(model: str) -> bool:
        return "multimodal" in model.lower()


def _convert_content_block(block: dict) -> dict:
    """Convert an OpenAI-style content block to Voyage multimodal format."""
    block_type = block.get("type", "")

    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}

    if block_type == "image_url":
        url = block.get("image_url", "")
        if isinstance(url, dict):
            url = url.get("url", "")
        if url.startswith("data:"):
            return {"type": "image_base64", "image_base64": url}
        return {"type": "image_url", "image_url": url}

    if block_type == "image_base64":
        return {
            "type": "image_base64",
            "image_base64": block.get("image_base64", ""),
        }

    if block_type == "video_url":
        return {"type": "video_url", "video_url": block.get("video_url", "")}

    if block_type == "video_base64":
        return {
            "type": "video_base64",
            "video_base64": block.get("video_base64", ""),
        }

    return block
