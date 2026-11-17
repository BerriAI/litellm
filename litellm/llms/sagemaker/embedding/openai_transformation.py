"""
Translate from OpenAI's `/v1/embeddings` to SageMaker's `/invoke`

For endpoints whose container already serves the OpenAI embeddings API, opted into per
deployment with the `openai/` route (`sagemaker/openai/<endpoint-name>`): the request
goes out as `{"input": [...]}` plus the caller's extra body fields, and the response is
read as the OpenAI embeddings object (`data[].embedding`, `usage`).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import STR_KEYED_MAPPING, SagemakerError

SUPPORTED_OPENAI_PARAMS: Final = ("dimensions", "encoding_format", "user")
_DECODED_JSON: Final = TypeAdapter(object)


class OpenAIEmbeddingUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int = 0
    total_tokens: int | None = None


class OpenAIEmbeddingRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    embedding: tuple[float, ...] | str
    index: int | None = None


class OpenAIEmbeddingsPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: tuple[OpenAIEmbeddingRow, ...]
    model: str | None = None
    usage: OpenAIEmbeddingUsage | None = None


class SagemakerOpenAIEmbeddingConfig(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseConfig returns a list
        return list(SUPPORTED_OPENAI_PARAMS)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: callers keep adding to the mapped params
        mapped: Final = {k: v for k, v in non_default_params.items() if k in SUPPORTED_OPENAI_PARAMS}
        return {**optional_params, **mapped}

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | Headers,  # mutable-ok: SagemakerError takes the same header types
    ) -> BaseLLMException:
        return SagemakerError(message=error_message, status_code=status_code, headers=headers)

    def transform_embedding_request(
        self,
        model: str,
        input: "AllEmbeddingInputValues",
        optional_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: BaseEmbeddingConfig returns the request body as a dict
        extra_body: Final = optional_params.get("extra_body")
        extra_fields: Final[Mapping[str, object]] = (
            STR_KEYED_MAPPING.validate_python(extra_body) if isinstance(extra_body, dict) else {}
        )
        top_level_params: Final = {k: v for k, v in optional_params.items() if k != "extra_body"}
        return {"input": input, **top_level_params, **extra_fields}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: "EmbeddingResponse",
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None = None,
        request_data: Mapping[str, object] | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> "EmbeddingResponse":
        try:
            response_data: Final = _DECODED_JSON.validate_python(raw_response.json())
        except ValueError as e:
            raise SagemakerError(
                message=f"Failed to parse response: {e}",
                status_code=raw_response.status_code,
            )

        try:
            payload: Final = OpenAIEmbeddingsPayload.model_validate(response_data)
        except ValidationError as e:
            raise SagemakerError(
                status_code=500,
                message=(
                    "Unexpected response format. Expected an OpenAI embeddings object with a 'data' list of "
                    f"{{'embedding': [...]}} rows: {e}"
                ),
            )

        prompt_tokens: Final = payload.usage.prompt_tokens if payload.usage else 0
        total_tokens: Final = (
            payload.usage.total_tokens if payload.usage and payload.usage.total_tokens is not None else prompt_tokens
        )

        return model_response.model_copy(
            update={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": row.index if row.index is not None else idx,
                        "embedding": row.embedding if isinstance(row.embedding, str) else list(row.embedding),
                    }
                    for idx, row in enumerate(payload.data)
                ],
                "model": payload.model or model,
                "usage": Usage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=total_tokens),
            }
        )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseConfig returns the request headers as a dict
        return {"Content-Type": "application/json"}
