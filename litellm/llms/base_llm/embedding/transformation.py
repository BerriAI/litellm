import base64
import struct
from abc import ABC, abstractmethod
from collections.abc import Mapping
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


_BODY_FIELDS: Final = TypeAdapter(dict[str, object])


def extra_body_fields(optional_params: Mapping[str, object] | None) -> Mapping[str, object]:
    """The caller's ``extra_body`` fields, merged under the provider body after the transform so they add fields but never
    replace the input or a mapped param."""
    extra_body: Final = (optional_params or {}).get("extra_body")
    return _BODY_FIELDS.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}


def split_extra_body(
    optional_params: Mapping[str, object] | None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """The params without ``extra_body``, and the ``extra_body`` fields to merge into the provider body."""
    params: Final = _BODY_FIELDS.validate_python(optional_params or {})
    return {k: v for k, v in params.items() if k != "extra_body"}, extra_body_fields(params)


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
