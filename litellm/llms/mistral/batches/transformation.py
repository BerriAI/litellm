"""
Mistral Batch API. Reference: https://docs.mistral.ai/api/#tag/batch

Mistral runs one model per job (set on the job, not per input line) and accepts
``/v1/ocr`` as a batch endpoint, which is how OCR gets its 50% batch discount.
Output and error files are OpenAI-shaped JSONL (``{custom_id, response: {status_code, body}}``),
so the shared batch cost accounting reads them without a provider branch.

Job lists are paged by a 0-based ``page`` number rather than an id cursor, so ``after`` carries
the page number the previous response handed back as ``next_page_token``.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlencode

import httpx
from openai.types.batch import BatchRequestCounts
from openai.types.batch import Errors as BatchErrors
from openai.types.batch_error import BatchError
from pydantic import BaseModel, ConfigDict
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.batches.transformation import (
    BaseBatchesCancelConfig,
    BaseBatchesListConfig,
    BatchHttpRequest,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, CreateBatchRequest
from litellm.types.utils import LiteLLMBatch, LlmProviders, OpenAIBatchListResponse

from ..common_utils import get_mistral_api_base, get_mistral_auth_headers, mistral_error

MistralBatchStatus: TypeAlias = Literal[
    "QUEUED", "RUNNING", "SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLATION_REQUESTED", "CANCELLED"
]
OpenAIBatchStatus: TypeAlias = Literal[
    "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
]

OPENAI_LIST_DEFAULT_LIMIT: Final = 20
_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})
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


class MistralListBatchJobsQuery(TypedDict):
    """Query of ``GET /v1/batch/jobs``."""

    page: ReadOnly[int]
    page_size: ReadOnly[int]


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


class MistralBatchJobList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    total: int
    data: tuple[MistralBatchJob, ...] = ()


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


def _page_number(after: str | None) -> int:
    if after is None:
        return 0
    if not after.isdecimal():
        raise mistral_error(
            f"Mistral pages batch jobs by number: pass the previous page's next_page_token as 'after', got {after!r}",
            400,
            _NO_HEADERS,
        )
    return int(after)


def _batch_http_request(
    method: Literal["GET", "POST"], path: str, litellm_params: Mapping[str, object]
) -> BatchHttpRequest:
    api_base: Final = litellm_params.get("api_base")
    api_key: Final = litellm_params.get("api_key")
    return BatchHttpRequest(
        method=method,
        url=f"{get_mistral_api_base(api_base if isinstance(api_base, str) else None)}{path}",
        headers=get_mistral_auth_headers(_NO_HEADERS, api_key if isinstance(api_key, str) else None),
    )


class MistralBatchesConfig(BaseBatchesListConfig, BaseBatchesCancelConfig):
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
        request: Final = _batch_http_request("GET", f"/v1/batch/jobs/{encoded_batch_id}", litellm_params)
        return dict(request)  # mutable-ok: BaseBatchesConfig signature

    def transform_retrieve_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: object,
        litellm_params: Mapping[str, object],
    ) -> LiteLLMBatch:
        return _to_litellm_batch(MistralBatchJob.model_validate(raw_response.json()))

    def transform_list_batches_request(
        self,
        after: str | None,
        limit: int | None,
        litellm_params: Mapping[str, object],
    ) -> BatchHttpRequest:
        query: Final = MistralListBatchJobsQuery(
            page=_page_number(after),
            page_size=limit if limit is not None else OPENAI_LIST_DEFAULT_LIMIT,
        )
        return _batch_http_request("GET", f"/v1/batch/jobs?{urlencode(query)}", litellm_params)

    def transform_list_batches_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: object,
        litellm_params: Mapping[str, object],
    ) -> OpenAIBatchListResponse:
        page: Final = MistralBatchJobList.model_validate(raw_response.json())
        sent_query: Final = raw_response.request.url.params
        page_number: Final = int(sent_query["page"])
        page_size: Final = int(sent_query["page_size"])
        data: Final = tuple(_to_litellm_batch(job) for job in page.data)
        has_more: Final = (page_number + 1) * page_size < page.total
        return OpenAIBatchListResponse(
            data=data,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
            has_more=has_more,
            next_page_token=str(page_number + 1) if has_more else None,
        )

    def transform_cancel_batch_request(
        self,
        batch_id: str,
        litellm_params: Mapping[str, object],
    ) -> BatchHttpRequest:
        encoded_batch_id: Final = encode_url_path_segment(batch_id, field_name="batch_id")
        return _batch_http_request("POST", f"/v1/batch/jobs/{encoded_batch_id}/cancel", litellm_params)

    def transform_cancel_batch_response(
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
