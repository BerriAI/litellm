"""
This module is used to transform the request and response for the Voyage contextualized embeddings API.
This would be used for all the contextualized embeddings models in Voyage.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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
        self.request = httpx.Request(method="POST", url="https://api.voyageai.com/v1/contextualizedembeddings")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class VoyageContextualEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.voyageai.com/reference/embeddings-api
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
            if not api_base.endswith("/contextualizedembeddings"):
                api_base = f"{api_base}/contextualizedembeddings"
            return api_base
        return "https://api.voyageai.com/v1/contextualizedembeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Voyage params

        Reference: https://docs.voyageai.com/reference/contextualized-embeddings-api
        """
        if "encoding_format" in non_default_params:
            optional_params["encoding_format"] = non_default_params["encoding_format"]
        if "dimensions" in non_default_params:
            optional_params["output_dimension"] = non_default_params["dimensions"]
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

    # Chunk size (in tokens) used when the API auto-chunks a flat ``list[str]``.
    # Matches the voyage-context-4 context window so each string stays a single
    # chunk instead of being split.
    AUTO_CHUNK_SIZE = 32000

    def transform_embedding_request(
        self,
        model: str,
        input: Union[AllEmbeddingInputValues, List[List[str]]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        inputs, extra_params = self._transform_contextual_inputs(input, optional_params)
        return {
            "inputs": inputs,
            "model": model,
            **optional_params,
            **extra_params,
        }

    @classmethod
    def _transform_contextual_inputs(
        cls,
        input: Union[AllEmbeddingInputValues, List[List[str]]],
        optional_params: dict,
    ) -> Tuple[Union[List[str], List[List[str]]], dict]:
        """
        Normalize ``input`` for Voyage's contextualized embeddings API and
        return ``(inputs, extra_params)`` where ``extra_params`` carries any
        request fields (e.g. auto-chunking) needed for the chosen shape.

        The API contract (verified against the live endpoint) is:

        - A flat ``list[str]`` is only accepted with ``input_type="query"`` or
          with ``enable_auto_chunking=True`` (which itself requires
          ``input_type="document"``).
        - A ``list[list[str]]`` (each inner list = one document's chunks) is
          always accepted.

        So we prefer to send a flat ``list[str]`` and let the API auto-chunk,
        instead of pre-wrapping into ``list[list[str]]``:

        - ``str`` -> ``[str]`` + ``enable_auto_chunking`` (input_type=document)
        - flat ``list[str]`` + ``input_type="query"`` -> kept flat, as-is
        - flat ``list[str]`` otherwise -> kept flat + ``enable_auto_chunking``
          (input_type=document)
        - ``list[list[str]]`` -> passed through unchanged

        Reference: https://docs.voyageai.com/reference/contextualized-embeddings-api
        """
        # Single string -> a one-element flat list, auto-chunked.
        if isinstance(input, str):
            if optional_params.get("input_type") == "query":
                return [input], {}
            return [input], cls._auto_chunk_params(optional_params)

        # Flat list[str].
        if isinstance(input, list) and all(isinstance(i, str) for i in input):
            if optional_params.get("input_type") == "query":
                # The API accepts a flat query list as-is.
                return input, {}  # type: ignore[return-value]
            # Otherwise let the API auto-chunk the flat list.
            return input, cls._auto_chunk_params(optional_params)  # type: ignore[return-value]

        # Already list[list[str]] (or another shape) -> pass through unchanged.
        return input, {}  # type: ignore[return-value]

    @classmethod
    def _auto_chunk_params(cls, optional_params: dict) -> dict:
        """
        Params required to send a flat ``list[str]`` to the contextualized API.

        ``enable_auto_chunking=True`` requires ``input_type="document"``, so set
        it unless the caller already provided an ``input_type``.
        """
        params: Dict[str, Any] = {
            "enable_auto_chunking": True,
            "chunk_size": cls.AUTO_CHUNK_SIZE,
        }
        if not optional_params.get("input_type"):
            params["input_type"] = "document"
        return params

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
            raise VoyageError(message=raw_response.text, status_code=raw_response.status_code)

        # model_response.usage
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
        return VoyageError(message=error_message, status_code=status_code, headers=headers)

    @staticmethod
    def is_contextualized_embeddings(model: str) -> bool:
        return "context" in model.lower()
