"""
Mistral Batch API. Reference: https://docs.mistral.ai/api/#tag/batch

Mistral runs one model per job (set on the job, not per input line) and accepts
``/v1/ocr`` as a batch endpoint, which is how OCR gets its 50% batch discount.
Output and error files are OpenAI-shaped JSONL (``{custom_id, response: {status_code, body}}``),
so the shared batch cost accounting reads them without a provider branch.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from openai.types.batch import BatchRequestCounts
from openai.types.batch import Errors as BatchErrors
from openai.types.batch_error import BatchError
from pydantic import BaseModel, ConfigDict
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, CreateBatchRequest
from litellm.types.utils import LiteLLMBatch, LlmProviders

from ..common_utils import get_mistral_api_base, get_mistral_auth_headers, mistral_error

MistralBatchStatus: TypeAlias = Literal[
    "QUEUED", "RUNNING", "SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLATION_REQUESTED", "CANCELLED"
]
OpenAIBatchStatus: TypeAlias = Literal[
    "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
]

_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})  # mutable-ok: frozen at module scope
_STATUS_MAP: Final[MappingProxyType[MistralBatchStatus, OpenAIBatchStatus]] = MappingProxyType(
    {
        "QUEUED": "validating",
        "RUNNING": "in_progress",
        "SUCCESS": "completed",
        "FAILED": "failed",
        "TIMEOUT_EXCEEDED": "expired",
        "CANCELLATION_REQUESTED": "cancelling",
        "CANCELLED": "cancelled",
    }
)


class MistralCreateBatchJobRequest(TypedDict):
    """Body of ``POST /v1/batch/jobs``."""

    input_files: ReadOnly[tuple[str, ...]]
    endpoint: ReadOnly[str]
    model: ReadOnly[str]
    metadata: NotRequired[ReadOnly[Mapping[str, str]]]


class MistralPresignedRequest(TypedDict):
    """A fully-formed request the shared HTTP handler sends as-is (its ``method`` branch)."""

    method: ReadOnly[Literal["GET"]]
    url: ReadOnly[str]
    headers: ReadOnly[Mapping[str, str]]


class MistralBatchError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    message: str
    count: int = 1


class MistralBatchJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    input_files: tuple[str, ...] = ()
    endpoint: str
    model: str | None = None
    status: MistralBatchStatus
    created_at: int
    started_at: int | None = None
    completed_at: int | None = None
    total_requests: int = 0
    completed_requests: int = 0
    succeeded_requests: int = 0
    failed_requests: int = 0
    output_file: str | None = None
    error_file: str | None = None
    errors: tuple[MistralBatchError, ...] = ()
    metadata: dict[str, str] | None = None  # mutable-ok: LiteLLMBatch.metadata is typed as dict


def _to_batch_errors(errors: Sequence[MistralBatchError]) -> BatchErrors | None:
    if not errors:
        return None
    return BatchErrors(
        object="list",
        data=[  # mutable-ok: openai Batch.Errors.data is typed as list
            BatchError(message=f"{e.message} (x{e.count})" if e.count > 1 else e.message) for e in errors
        ],
    )


def _to_litellm_batch(job: MistralBatchJob) -> LiteLLMBatch:
    status: Final = _STATUS_MAP[job.status]
    terminal_at: Final = job.completed_at
    return LiteLLMBatch(
        id=job.id,
        object="batch",
        endpoint=job.endpoint,
        input_file_id=job.input_files[0] if job.input_files else "",
        completion_window="24h",
        status=status,
        created_at=job.created_at,
        in_progress_at=job.started_at,
        completed_at=terminal_at if status == "completed" else None,
        failed_at=terminal_at if status == "failed" else None,
        expired_at=terminal_at if status == "expired" else None,
        cancelled_at=terminal_at if status == "cancelled" else None,
        output_file_id=job.output_file,
        error_file_id=job.error_file,
        errors=_to_batch_errors(job.errors),
        request_counts=BatchRequestCounts(
            total=job.total_requests,
            completed=job.succeeded_requests,
            failed=job.failed_requests,
        ),
        metadata=job.metadata,
    )


class MistralBatchesConfig(BaseBatchesConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.MISTRAL

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
        return get_mistral_auth_headers(headers, api_key)

    def get_complete_batch_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        data: CreateBatchRequest,
    ) -> str:
        return f"{get_mistral_api_base(api_base)}/v1/batch/jobs"

    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseBatchesConfig signature
        input_file_id: Final = create_batch_data.get("input_file_id")
        endpoint: Final = create_batch_data.get("endpoint")
        if input_file_id is None or endpoint is None:
            raise ValueError("input_file_id and endpoint are required to create a Mistral batch job")
        metadata: Final = create_batch_data.get("metadata")
        body: Final = (
            MistralCreateBatchJobRequest(
                input_files=(input_file_id,), endpoint=endpoint, model=model, metadata=metadata
            )
            if metadata
            else MistralCreateBatchJobRequest(input_files=(input_file_id,), endpoint=endpoint, model=model)
        )
        return dict(body)  # mutable-ok: BaseBatchesConfig signature

    def transform_create_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: object,
        litellm_params: Mapping[str, object],
    ) -> LiteLLMBatch:
        return _to_litellm_batch(MistralBatchJob.model_validate(raw_response.json()))

    def transform_retrieve_batch_request(
        self,
        batch_id: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseBatchesConfig signature
        encoded_batch_id: Final = encode_url_path_segment(batch_id, field_name="batch_id")
        api_base: Final = litellm_params.get("api_base")
        api_key: Final = litellm_params.get("api_key")
        request: Final = MistralPresignedRequest(
            method="GET",
            url=f"{get_mistral_api_base(api_base if isinstance(api_base, str) else None)}/v1/batch/jobs/{encoded_batch_id}",
            headers=get_mistral_auth_headers(_NO_HEADERS, api_key if isinstance(api_key, str) else None),
        )
        return dict(request)  # mutable-ok: BaseBatchesConfig signature

    def transform_retrieve_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: object,
        litellm_params: Mapping[str, object],
    ) -> LiteLLMBatch:
        return _to_litellm_batch(MistralBatchJob.model_validate(raw_response.json()))

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return mistral_error(error_message, status_code, headers)
