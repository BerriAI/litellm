"""
This module is used to transform the request and response for the Voyage multimodal embeddings API.
This would be used for voyage-multimodal-3 and voyage-multimodal-3.5 models.

Reference: https://docs.voyageai.com/docs/multimodal-embeddings
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class VoyageMultimodalError(BaseLLMException):
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
    Configuration for Voyage multimodal embeddings API.

    Reference: https://docs.voyageai.com/docs/multimodal-embeddings

    Supported input formats:
    1. Simple text strings: ["text1", "text2"]
    2. OpenAI-like format with content arrays:
       [{"content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": "..."}]}]
    3. Mixed content lists for each input
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
        return [
            "encoding_format",
            "dimensions",
            "input_type",
            "truncation",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Voyage multimodal params.

        Reference: https://docs.voyageai.com/docs/multimodal-embeddings
        """
        if "encoding_format" in non_default_params:
            optional_params["encoding_format"] = non_default_params["encoding_format"]
        if "dimensions" in non_default_params:
            optional_params["output_dimension"] = non_default_params["dimensions"]
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
        }

    def _convert_input_to_multimodal_format(
        self, input_item: Any
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Convert a single input item to Voyage multimodal format.

        Voyage multimodal API expects:
        {
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": "..."},
                {"type": "image_base64", "image_base64": "..."},
                {"type": "video_url", "video_url": "..."},
                {"type": "video_base64", "video_base64": "..."}
            ]
        }

        Supported input formats:
        1. Plain text string -> {"content": [{"type": "text", "text": "..."}]}
        2. Dict with "content" key (explicit format) -> pass through
        3. List of content dicts -> wrap in {"content": [...]}
        """
        # Already in the correct format with "content" key
        if isinstance(input_item, dict) and "content" in input_item:
            return input_item

        # Plain text string
        if isinstance(input_item, str):
            return {"content": [{"type": "text", "text": input_item}]}

        # List of content elements with explicit type
        # e.g., [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": "..."}]
        if isinstance(input_item, list):
            content_list = []
            for element in input_item:
                if isinstance(element, str):
                    # Plain text string in list
                    content_list.append({"type": "text", "text": element})
                elif isinstance(element, dict):
                    # Already a content object with type
                    if "type" in element:
                        content_list.append(element)
                    # OpenAI-style image_url format: {"image_url": {"url": "..."}}
                    elif "image_url" in element:
                        url = element["image_url"]
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        content_list.append({"type": "image_url", "image_url": url})
                    elif "text" in element:
                        content_list.append({"type": "text", "text": element["text"]})
                    else:
                        # Unknown dict format, skip or raise?
                        pass
            return {"content": content_list}

        # Fallback: treat as text
        return {"content": [{"type": "text", "text": str(input_item)}]}

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the embedding request to Voyage multimodal format.

        Voyage multimodal API expects:
        {
            "inputs": [
                {"content": [{"type": "text", "text": "..."}, ...]},
                {"content": [{"type": "image_url", "image_url": "..."}, ...]}
            ],
            "model": "voyage-multimodal-3.5",
            "input_type": "query" | "document",  # optional
            "truncation": true,  # optional
            "output_dimension": 1024  # optional
        }
        """
        # Convert input to list if it's a single item
        input_list: List[Any]
        if isinstance(input, str):
            input_list = [input]
        elif isinstance(input, list):
            input_list = input
        else:
            input_list = [input]

        # Convert each input to multimodal format
        multimodal_inputs = [
            self._convert_input_to_multimodal_format(item) for item in input_list
        ]

        return {
            "inputs": multimodal_inputs,
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
            raise VoyageMultimodalError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        # Voyage multimodal returns usage with text_tokens, image_pixels, total_tokens
        usage_data = raw_response_json.get("usage", {})
        total_tokens = usage_data.get("total_tokens", 0)
        text_tokens = usage_data.get("text_tokens", 0)

        usage = Usage(
            prompt_tokens=text_tokens if text_tokens else total_tokens,
            total_tokens=total_tokens,
        )
        model_response.usage = usage
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return VoyageMultimodalError(
            message=error_message, status_code=status_code, headers=headers
        )

    @staticmethod
    def is_multimodal_embedding(model: str) -> bool:
        """Check if the model is a multimodal embedding model."""
        return "multimodal" in model.lower()
