from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.gpustack.common_utils import (
    get_gpustack_endpoint,
    get_gpustack_headers,
    strip_gpustack_model_prefix,
)
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse


class GPUStackEmbeddingError(BaseLLMException):
    pass


# fmt: off
class GPUStackEmbeddingConfig(BaseEmbeddingConfig):
    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        optional_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        litellm_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        return get_gpustack_headers(headers=headers, api_key=api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        litellm_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        stream: bool | None = None,
    ) -> str:
        return get_gpustack_endpoint(api_base=api_base, endpoint="/embeddings")

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        headers: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
    ) -> dict[str, object]:  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        encoding_format: Final[object | None] = optional_params.get("encoding_format")
        encoding_format_body: Final = (
            {"encoding_format": encoding_format} if encoding_format not in (None, "") else {}  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        )  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        return {  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
            "model": strip_gpustack_model_prefix(model),
            "input": input,
            **encoding_format_body,
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: object,
        api_key: str | None,
        request_data: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        optional_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        litellm_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
    ) -> EmbeddingResponse:
        return TypeAdapter(EmbeddingResponse).validate_json(raw_response.content)

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        return ["encoding_format", "timeout"]  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers

    def map_openai_params(
        self,
        non_default_params: dict[  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
            str, object
        ],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        optional_params: dict[str, object],  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        return {  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
            **optional_params,
            **{  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
                param: value for param, value in non_default_params.items() if param == "encoding_format"
            },  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object]  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
        | httpx.Headers,  # mutable-ok: LiteLLM provider interfaces require mutable JSON containers
    ) -> BaseLLMException:
        return GPUStackEmbeddingError(message=error_message, status_code=status_code, headers=headers)
# fmt: on
