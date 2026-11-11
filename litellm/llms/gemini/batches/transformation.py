"""
Gemini Developer API batch jobs. Reference: https://ai.google.dev/gemini-api/docs/batch-api

A file-backed job runs the uploaded JSONL unchanged. Gemini accepts OpenAI-shaped rows
(``{custom_id, method, url, body}``) and then writes the output file OpenAI-shaped too
(``{custom_id, response: {status_code, body}}``), so ``/v1/files purpose=batch`` uploads and the
shared batch cost accounting need no provider branch. The model is set on the job URL.
Ids cross the LiteLLM boundary bare: Gemini's ``batches/<id>`` is the batch id ``<id>`` and its
``files/<id>`` output file is the file id ``<id>``.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from openai.types.batch import BatchRequestCounts
from openai.types.batch import Errors as BatchErrors
from openai.types.batch_error import BatchError
from pydantic import ConfigDict, Field
from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.url_utils import encode_url_path_segment, join_api_base
from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.gemini.common_utils import GeminiError, GeminiModelInfo
from litellm.llms.gemini.files.transformation import gemini_file_name
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import AllMessageValues, CreateBatchRequest
from litellm.types.utils import LiteLLMBatch, LlmProviders, Usage

GeminiBatchState: TypeAlias = Literal[
    "BATCH_STATE_UNSPECIFIED",
    "BATCH_STATE_PENDING",
    "BATCH_STATE_RUNNING",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
]
OpenAIBatchStatus: TypeAlias = Literal[
    "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
]

GEMINI_BATCH_RESOURCE_PREFIX: Final = "batches/"
_GEMINI_FILE_RESOURCE_PREFIX: Final = "files/"
_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})
_STATUS_MAP: Final[MappingProxyType[GeminiBatchState, OpenAIBatchStatus]] = MappingProxyType(
    {
        "BATCH_STATE_UNSPECIFIED": "validating",
        "BATCH_STATE_PENDING": "validating",
        "BATCH_STATE_RUNNING": "in_progress",
        "BATCH_STATE_SUCCEEDED": "completed",
        "BATCH_STATE_FAILED": "failed",
        "BATCH_STATE_CANCELLED": "cancelled",
        "BATCH_STATE_EXPIRED": "expired",
    }
)


class GeminiBatchInputConfig(TypedDict):
    file_name: ReadOnly[str]


class GeminiBatchSpec(TypedDict):
    display_name: ReadOnly[str]
    input_config: ReadOnly[GeminiBatchInputConfig]


class GeminiCreateBatchRequest(TypedDict):
    batch: ReadOnly[GeminiBatchSpec]


class GeminiPresignedRequest(TypedDict):
    method: ReadOnly[Literal["GET"]]
    url: ReadOnly[str]
    headers: ReadOnly[Mapping[str, str]]


class _GeminiBatchStats(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    request_count: int = Field(default=0, alias="requestCount")
    successful_request_count: int = Field(default=0, alias="successfulRequestCount")
    failed_request_count: int = Field(default=0, alias="failedRequestCount")


class _GeminiBatchInputConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    file_name: str | None = Field(default=None, alias="fileName")


class _GeminiBatchOutput(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    responses_file: str | None = Field(default=None, alias="responsesFile")


class _GeminiBatchMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    display_name: str | None = Field(default=None, alias="displayName")
    input_config: _GeminiBatchInputConfig = Field(default_factory=_GeminiBatchInputConfig, alias="inputConfig")
    output: _GeminiBatchOutput = Field(default_factory=_GeminiBatchOutput)
    create_time: datetime = Field(alias="createTime")
    end_time: datetime | None = Field(default=None, alias="endTime")
    batch_stats: _GeminiBatchStats = Field(default_factory=_GeminiBatchStats, alias="batchStats")
    state: GeminiBatchState = "BATCH_STATE_UNSPECIFIED"


class _GeminiOperationError(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    code: int | None = None
    message: str = ""


class GeminiBatchOperation(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    metadata: _GeminiBatchMetadata
    error: _GeminiOperationError | None = None


class _GeminiBatchOutputUsage(LiteLLMBaseModel):
    """The ``usage`` block Gemini writes on an OpenAI-shaped batch output row. ``totalTokens``
    includes thinking tokens that ``completionTokens`` leaves out, and Gemini bills them as output."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_tokens: int = Field(default=0, alias="promptTokens")
    completion_tokens: int = Field(default=0, alias="completionTokens")
    total_tokens: int = Field(default=0, alias="totalTokens")


def is_native_gemini_batch_output_row(row: Mapping[str, object]) -> bool:
    """Native rows carry Gemini's ``key`` (and a ``response`` or an error ``status``) where OpenAI-shaped rows
    carry ``custom_id``, so the shape is known even when the first row failed."""
    return "key" in row and "custom_id" not in row


def is_gemini_batch_usage_shape(usage_object: Mapping[str, object]) -> bool:
    return any(key in usage_object for key in ("promptTokens", "completionTokens", "totalTokens"))


def usage_from_batch_output(usage_object: Mapping[str, object]) -> Usage:
    reported: Final = _GeminiBatchOutputUsage.model_validate(usage_object)
    reasoning_tokens: Final = max(reported.total_tokens - reported.prompt_tokens - reported.completion_tokens, 0)
    return Usage(
        prompt_tokens=reported.prompt_tokens,
        completion_tokens=reported.completion_tokens + reasoning_tokens,
        total_tokens=reported.total_tokens,
        reasoning_tokens=reasoning_tokens or None,
    )


def _bare_id(resource_name: str | None, prefix: str) -> str | None:
    return None if resource_name is None else resource_name.removeprefix(prefix)


def _to_batch_errors(error: _GeminiOperationError | None) -> BatchErrors | None:
    if error is None:
        return None
    return BatchErrors(
        object="list", data=[BatchError(code=str(error.code) if error.code else None, message=error.message)]
    )


def _to_litellm_batch(operation: GeminiBatchOperation) -> LiteLLMBatch:
    meta: Final = operation.metadata
    status: Final = _STATUS_MAP[meta.state]
    terminal_at: Final = int(meta.end_time.timestamp()) if meta.end_time is not None else None
    return LiteLLMBatch(
        id=_bare_id(operation.name, GEMINI_BATCH_RESOURCE_PREFIX) or operation.name,
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id=_bare_id(meta.input_config.file_name, _GEMINI_FILE_RESOURCE_PREFIX) or "",
        completion_window="24h",
        status=status,
        created_at=int(meta.create_time.timestamp()),
        completed_at=terminal_at if status == "completed" else None,
        failed_at=terminal_at if status == "failed" else None,
        expired_at=terminal_at if status == "expired" else None,
        cancelled_at=terminal_at if status == "cancelled" else None,
        output_file_id=_bare_id(meta.output.responses_file, _GEMINI_FILE_RESOURCE_PREFIX),
        errors=_to_batch_errors(operation.error),
        request_counts=BatchRequestCounts(
            total=meta.batch_stats.request_count,
            completed=meta.batch_stats.successful_request_count,
            failed=meta.batch_stats.failed_request_count,
        ),
        metadata={"display_name": meta.display_name} if meta.display_name else None,
    )


def _auth_headers(headers: Mapping[str, str], api_key: str | None) -> dict[str, str]:  # mutable-ok: header contract
    resolved_api_key: Final = GeminiModelInfo.get_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is required for Gemini batch operations")
    return {**headers, "x-goog-api-key": resolved_api_key}


class GeminiBatchesConfig(BaseBatchesConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.GEMINI

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseBatchesConfig signature
        return _auth_headers(headers, api_key)

    def get_complete_batch_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        data: CreateBatchRequest,
    ) -> str:
        model_segment: Final = encode_url_path_segment(model.removeprefix("models/"), field_name="model")
        return join_api_base(
            GeminiModelInfo.get_api_base(api_base), f"v1beta/models/{model_segment}:batchGenerateContent"
        )

    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseBatchesConfig signature
        input_file_id: Final = create_batch_data.get("input_file_id")
        if input_file_id is None:
            raise ValueError("input_file_id is required to create a Gemini batch job")
        file_name: Final = gemini_file_name(input_file_id)
        metadata: Final = create_batch_data.get("metadata") or {}
        unsupported_metadata: Final = sorted(set(metadata) - {"display_name"})
        if unsupported_metadata:
            raise ValueError(f"Gemini batch jobs carry no metadata besides display_name, got {unsupported_metadata}")
        display_name: Final = (
            metadata.get("display_name") or f"litellm-{file_name.removeprefix(_GEMINI_FILE_RESOURCE_PREFIX)}"
        )
        body: Final = GeminiCreateBatchRequest(
            batch=GeminiBatchSpec(display_name=display_name, input_config=GeminiBatchInputConfig(file_name=file_name))
        )
        return dict(body)

    def transform_create_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: object,
        litellm_params: Mapping[str, object],
    ) -> LiteLLMBatch:
        return _to_litellm_batch(GeminiBatchOperation.model_validate(raw_response.json()))

    def transform_retrieve_batch_request(
        self,
        batch_id: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseBatchesConfig signature
        bare_batch_id: Final = batch_id.removeprefix(GEMINI_BATCH_RESOURCE_PREFIX)
        encoded_batch_id: Final = encode_url_path_segment(bare_batch_id, field_name="batch_id")
        api_base: Final = litellm_params.get("api_base")
        api_key: Final = litellm_params.get("api_key")
        request: Final = GeminiPresignedRequest(
            method="GET",
            url=join_api_base(
                GeminiModelInfo.get_api_base(api_base if isinstance(api_base, str) else None),
                f"v1beta/{GEMINI_BATCH_RESOURCE_PREFIX}{encoded_batch_id}",
            ),
            headers=_auth_headers(_NO_HEADERS, api_key if isinstance(api_key, str) else None),
        )
        return dict(request)

    def transform_retrieve_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: object,
        litellm_params: Mapping[str, object],
    ) -> LiteLLMBatch:
        return _to_litellm_batch(GeminiBatchOperation.model_validate(raw_response.json()))

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return GeminiError(status_code=status_code, message=error_message, headers=dict(headers))
