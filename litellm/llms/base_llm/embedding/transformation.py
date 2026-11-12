from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_STR_KEYED_PARAMS: Final = TypeAdapter(dict[str, object])


def merge_extra_body(optional_params: Mapping[str, object]) -> dict[str, object]:
    """Lift ``extra_body`` fields to the top level, extra_body winning, as the OpenAI SDK does with its own body."""
    extra_body: Final = optional_params.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return dict(optional_params)
    top_level: Final = {k: v for k, v in optional_params.items() if k != "extra_body"}
    return {**top_level, **_STR_KEYED_PARAMS.validate_python(extra_body)}


class BaseEmbeddingConfig(BaseConfig, ABC):
    @abstractmethod
    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {}

    @abstractmethod
    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        return model_response

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError("EmbeddingConfig does not need a request transformation for chat models")

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        raise NotImplementedError("EmbeddingConfig does not need a response transformation for chat models")
