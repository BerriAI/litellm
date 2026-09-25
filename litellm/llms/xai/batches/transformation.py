"""
xAI Batch API reference: https://docs.x.ai/developers/advanced-api-usage/batch-api

xAI batches carry request counters, not a status, and no output file: results are paged from
``GET /v1/batches/{id}/results``, so LiteLLM hands back the batch id as ``output_file_id``.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from openai.types.batch import BatchRequestCounts
from openai.types.batch import Errors as BatchErrors
from openai.types.batch_error import BatchError
from pydantic import BaseModel, ConfigDict
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import XAI_API_BASE
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.utils import LiteLLMBatch

OpenAIBatchStatus: TypeAlias = Literal[
    "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
]

XAI_BATCH_ID_PREFIX: Final = "batch_"
XAI_RESULTS_PAGE_SIZE: Final = 1000
DEFAULT_BATCH_NAME: Final = "litellm-batch"
DEFAULT_BATCH_ENDPOINT: Final = "/v1/chat/completions"
_EMPTY_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


class XAIBatchesError(BaseLLMException):
    pass


def xai_batches_error(
    error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
) -> XAIBatchesError:
    return XAIBatchesError(
        status_code=status_code,
        message=error_message,
        headers=headers if isinstance(headers, httpx.Headers) else httpx.Headers(tuple(headers.items())),
    )


def raise_for_xai_status(response: httpx.Response) -> httpx.Response:
    if response.status_code >= 400:
        raise xai_batches_error(response.text, response.status_code, response.headers)
    return response


def get_xai_api_base(api_base: str | None) -> str:
    resolved: Final = (api_base or get_secret_str("XAI_API_BASE") or XAI_API_BASE).rstrip("/")
    return resolved.removesuffix("/v1")


def get_xai_auth_headers(
    headers: Mapping[str, str] = _EMPTY_HEADERS, api_key: str | None = None
) -> dict[str, str]:  # mutable-ok: BaseConfig.validate_environment contract returns dict
    resolved_key: Final = XAIModelInfo.get_api_key(api_key)
    if resolved_key is None:
        raise xai_batches_error(
            "Missing xAI API Key. Pass api_key, set litellm.xai_key or XAI_API_KEY", 401, _EMPTY_HEADERS
        )
    return dict(headers, Authorization=f"Bearer {resolved_key}")  # mutable-ok: BaseConfig contract returns dict


def xai_batches_url(api_base: str | None, batch_id: str | None = None, suffix: str = "") -> str:
    base: Final = f"{get_xai_api_base(api_base)}/v1/batches"
    if batch_id is None:
        return base
    return f"{base}/{encode_url_path_segment(batch_id, field_name='batch_id')}{suffix}"


def is_xai_batch_results_id(file_id: str) -> bool:
    return file_id.startswith(XAI_BATCH_ID_PREFIX)


class XAICreateBatchRequest(TypedDict):
    name: ReadOnly[str]
    input_file_id: NotRequired[ReadOnly[str]]


class XAIBatchState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    num_requests: int = 0
    num_pending: int = 0
    num_success: int = 0
    num_error: int = 0
    num_cancelled: int = 0


class XAIBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    batch_id: str
    name: str = ""
    create_time: str | None = None
    expire_time: str | None = None
    cancel_time: str | None = None
    cancel_by_xai_message: str | None = None
    state: XAIBatchState = XAIBatchState()
    input_file_id: str | None = None


class XAIBatchList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    batches: tuple[XAIBatch, ...] = ()
    pagination_token: str | None = None


class XAIBatchResultError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    code: int | str | None = None
    message: str = ""


class XAIBatchResultData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    response: Mapping[str, Mapping[str, object]] | None = None
    error: XAIBatchResultError | None = None


class XAIBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    batch_request_id: str
    batch_result: XAIBatchResultData = XAIBatchResultData()


class XAIBatchResultsPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    results: tuple[XAIBatchResult, ...] = ()
    pagination_token: str | None = None


def _to_unix_timestamp(value: str | None) -> int | None:
    """xAI returns RFC 3339 timestamps over gRPC but a bare ``YYYY-MM-DD`` over REST."""
    if value is None:
        return None
    try:
        parsed: Final = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int((parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)).timestamp())


def xai_batch_status(batch: XAIBatch) -> OpenAIBatchStatus:
    """xAI exposes counters, not a status. A batch xAI itself cancelled (input validation failed) is a failure,
    a caller-cancelled batch is cancelled, an empty batch is still validating its input file, and a batch
    with nothing pending has completed."""
    if batch.cancel_time is not None:
        return "failed" if batch.cancel_by_xai_message else "cancelled"
    if batch.state.num_requests == 0:
        return "validating"
    if batch.state.num_pending > 0:
        return "in_progress"
    return "completed"


def to_litellm_batch(batch: XAIBatch, endpoint: str = DEFAULT_BATCH_ENDPOINT) -> LiteLLMBatch:
    status: Final = xai_batch_status(batch)
    created_at: Final = _to_unix_timestamp(batch.create_time)
    cancelled_at: Final = _to_unix_timestamp(batch.cancel_time)
    errors: Final = (
        BatchErrors(object="list", data=[BatchError(message=batch.cancel_by_xai_message)])  # mutable-ok: openai type
        if batch.cancel_by_xai_message
        else None
    )
    return LiteLLMBatch(
        id=batch.batch_id,
        object="batch",
        endpoint=endpoint,
        input_file_id=batch.input_file_id or "",
        completion_window="24h",
        status=status,
        created_at=created_at if created_at is not None else 0,
        expires_at=_to_unix_timestamp(batch.expire_time),
        failed_at=cancelled_at if status == "failed" else None,
        cancelled_at=cancelled_at if status == "cancelled" else None,
        output_file_id=batch.batch_id if status == "completed" else None,
        errors=errors,
        request_counts=BatchRequestCounts(
            total=batch.state.num_requests,
            completed=batch.state.num_success,
            failed=batch.state.num_error + batch.state.num_cancelled,
        ),
        metadata={"name": batch.name} if batch.name else None,  # mutable-ok: LiteLLMBatch.metadata is a dict
    )


class OpenAIBatchListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    object: Literal["list"] = "list"
    data: tuple[LiteLLMBatch, ...]
    first_id: str | None
    last_id: str | None
    has_more: bool
    next_page_token: str | None = None


def to_openai_batch_list(page: XAIBatchList) -> OpenAIBatchListResponse:
    data: Final = tuple(to_litellm_batch(b) for b in page.batches)
    return OpenAIBatchListResponse(
        data=data,
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
        has_more=bool(page.pagination_token),
        next_page_token=page.pagination_token or None,
    )


def to_create_batch_body(create_batch_data: CreateBatchRequest) -> XAICreateBatchRequest:
    input_file_id: Final = create_batch_data.get("input_file_id")
    if not input_file_id:
        raise xai_batches_error("input_file_id is required to create an xAI batch", 400, _EMPTY_HEADERS)
    metadata: Final = create_batch_data.get("metadata")
    name: Final = metadata.get("name") if metadata else None
    return XAICreateBatchRequest(name=name or DEFAULT_BATCH_NAME, input_file_id=input_file_id)


class OpenAIBatchOutputError(TypedDict):
    code: ReadOnly[str]
    message: ReadOnly[str]


class OpenAIBatchOutputResponse(TypedDict):
    status_code: ReadOnly[int]
    request_id: ReadOnly[object]
    body: ReadOnly[Mapping[str, object]]


class OpenAIBatchOutputLine(TypedDict):
    id: ReadOnly[str]
    custom_id: ReadOnly[str]
    response: ReadOnly[OpenAIBatchOutputResponse | None]
    error: ReadOnly[OpenAIBatchOutputError | None]


def _result_to_openai_line(result: XAIBatchResult) -> OpenAIBatchOutputLine:
    """One output JSONL line. xAI wraps the body in a one-key map named after the endpoint
    (``chat_get_completion``, ``responses``, ``image_generation``, ...); the value is the OpenAI body."""
    error: Final = result.batch_result.error
    response: Final = result.batch_result.response
    body: Final = next(iter(response.values()), None) if response else None
    if body is None:
        message: Final = error.message if error is not None else "xAI returned no response for this request"
        code: Final = str(error.code) if error is not None and error.code is not None else "request_failed"
        return OpenAIBatchOutputLine(
            id=f"batch_req_{result.batch_request_id}",
            custom_id=result.batch_request_id,
            response=None,
            error=OpenAIBatchOutputError(code=code, message=message),
        )
    return OpenAIBatchOutputLine(
        id=f"batch_req_{result.batch_request_id}",
        custom_id=result.batch_request_id,
        response=OpenAIBatchOutputResponse(status_code=200, request_id=body.get("id"), body=body),
        error=None,
    )


def results_to_openai_jsonl(results: Sequence[XAIBatchResult]) -> bytes:
    return "".join(f"{json.dumps(_result_to_openai_line(r), ensure_ascii=False)}\n" for r in results).encode()
