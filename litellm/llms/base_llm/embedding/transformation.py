import base64
import struct
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import Embedding, EmbeddingResponse, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class _EmbeddingItems(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    data: tuple[Embedding, ...]


class _EmbeddingVector(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    embedding: tuple[float, ...] | str


_EMBEDDING_VECTORS: Final = TypeAdapter(tuple[_EmbeddingVector, ...])


def _float32_base64(vector: tuple[float, ...] | str) -> str:
    if isinstance(vector, str):
        return vector
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def with_base64_embeddings(response: EmbeddingResponse) -> EmbeddingResponse:
    items: Final = _EmbeddingItems.model_validate(response, from_attributes=True).data
    vectors: Final = _EMBEDDING_VECTORS.validate_python(items, from_attributes=True)
    return response.model_copy(
        update={
            "data": [
                item.model_copy(update={"embedding": _float32_base64(vector.embedding)})
                for item, vector in zip(items, vectors, strict=True)
            ]
        }
    )


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
