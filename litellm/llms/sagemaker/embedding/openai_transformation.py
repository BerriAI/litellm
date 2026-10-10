"""
Translate from OpenAI's `/v1/embeddings` to SageMaker's `/invoke`

For endpoints whose container already serves the OpenAI embeddings API, opted into per
deployment with the `openai/` route (`sagemaker/openai/<endpoint-name>`): the request
goes out as `{"input": [...]}` plus the caller's extra body fields, and the response is
read as the OpenAI embeddings object (`data[].embedding`, `usage`).
"""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import AllEmbeddingInputValues

from httpx._models import Headers, Response
from pydantic import BaseModel, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import SagemakerError

SUPPORTED_OPENAI_PARAMS: Final = ("dimensions", "encoding_format", "user")


class OpenAIEmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int | None = None


class OpenAIEmbeddingRow(BaseModel):
    embedding: list[float] | str
    index: int | None = None


class OpenAIEmbeddingsPayload(BaseModel):
    data: list[OpenAIEmbeddingRow]
    model: str | None = None
    usage: OpenAIEmbeddingUsage | None = None


class SagemakerOpenAIEmbeddingConfig(BaseEmbeddingConfig):
    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> list[str]:
        return list(SUPPORTED_OPENAI_PARAMS)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped: Final = {k: v for k, v in non_default_params.items() if k in SUPPORTED_OPENAI_PARAMS}
        return {**optional_params, **mapped}

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return SagemakerError(message=error_message, status_code=status_code, headers=headers)

    def transform_embedding_request(
        self,
        model: str,
        input: "AllEmbeddingInputValues",
        optional_params: dict,
        headers: dict,
    ) -> dict:
        extra_body: Final = optional_params.get("extra_body")
        top_level_params: Final = {k: v for k, v in optional_params.items() if k != "extra_body"}
        return {
            "input": input,
            **top_level_params,
            **(extra_body if isinstance(extra_body, dict) else {}),
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: "EmbeddingResponse",
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None = None,
        request_data: dict | None = None,
        optional_params: dict | None = None,
        litellm_params: dict | None = None,
    ) -> "EmbeddingResponse":
        try:
            response_data: Final = raw_response.json()
        except Exception as e:
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

        model_response.object = "list"
        model_response.data = [
            {
                "object": "embedding",
                "index": row.index if row.index is not None else idx,
                "embedding": row.embedding,
            }
            for idx, row in enumerate(payload.data)
        ]
        model_response.model = payload.model or model
        model_response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=total_tokens)
        return model_response

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return {"Content-Type": "application/json"}
