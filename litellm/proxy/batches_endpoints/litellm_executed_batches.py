import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, runtime_checkable

import httpx
from openai.types.batch import Errors
from openai.types.batch_error import BatchError
from openai.types.batch_request_counts import BatchRequestCounts
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid as uuid_module
from litellm.constants import LITELLM_EXECUTED_BATCH_CONCURRENCY
from litellm.integrations.prometheus import PrometheusLogger
from litellm.llms.base_llm.files.litellm_db_storage_backend import LITELLM_DB_STORAGE_BACKEND_NAME
from litellm.llms.base_llm.files.storage_backend import BaseFileStorageBackend
from litellm.llms.base_llm.files.storage_backend_factory import get_storage_backend
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.models.managed_files import LiteLLM_ManagedFileTable
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.openai_files_endpoints.common_utils import (
    LITELLM_EXECUTED_BATCH_ID_PREFIX,
    convert_b64_uid_to_unified_uid,
    get_batch_id_from_unified_batch_id,
)
from litellm.proxy.openai_files_endpoints.storage_backend_service import StorageBackendFileService
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.table_repositories import ManagedObjectRepository
from litellm.types.llms.openai import LiteLLMBatchCreateRequest, OpenAIFileObject, OpenAIFilesPurpose
from litellm.types.utils import LITELLM_EXECUTED_BATCH_PROVIDERS, ExtractedFileData, LiteLLMBatch, LlmProviders

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.router import Router

BatchEndpoint: TypeAlias = Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"]
BatchStatus: TypeAlias = Literal["in_progress", "finalizing", "completed", "failed", "cancelling", "cancelled"]
TERMINAL_BATCH_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "cancelled", "expired"})
_STOP_STATUSES: Final[frozenset[str]] = TERMINAL_BATCH_STATUSES | frozenset({"cancelling"})
_BATCH_ENDPOINT_ADAPTER: Final[TypeAdapter[BatchEndpoint]] = TypeAdapter(BatchEndpoint)
_CANCEL_POLL_SECONDS: Final = 1.0
_HEARTBEAT_SECONDS: Final = 30.0
_STALE_AFTER_SECONDS: Final = 180.0
_FILES_API_PROBE_TIMEOUT_SECONDS: Final = 5.0
_COMPLETION_WINDOW_SECONDS: Final = 24 * 60 * 60
_RUNNER_LOST_MESSAGE: Final = "the proxy replica running this batch stopped before it finished; resubmit the batch"
_ROUTER_METHODS: Final[Mapping[BatchEndpoint, str]] = MappingProxyType(
    {
        "/v1/chat/completions": "acompletion",
        "/v1/completions": "atext_completion",
        "/v1/embeddings": "aembedding",
        "/v1/responses": "aresponses",
    }
)
_CANCELLING_TRANSITIONS: Final[Mapping[BatchStatus, BatchStatus]] = MappingProxyType(
    {"completed": "cancelled", "in_progress": "cancelling", "finalizing": "cancelling"}
)
LITELLM_EXECUTED_BATCH_UPLOAD_GUIDANCE: Final = (
    "upload it through POST /v1/files with purpose=batch and either the x-litellm-model header or the "
    "target_model_names form field naming the model, so LiteLLM keeps the file and runs the batch itself"
)
_RUNNING_BATCHES: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong references keep running batch tasks alive
_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


class _ErrorDetail(TypedDict):
    message: ReadOnly[str]
    type: ReadOnly[str]
    param: ReadOnly[None]
    code: ReadOnly[None]


class _ErrorBody(TypedDict):
    error: ReadOnly[_ErrorDetail]


class _ResultResponse(TypedDict):
    status_code: ReadOnly[int]
    request_id: ReadOnly[str]
    body: ReadOnly[Mapping[str, object]]


class _ResultLine(TypedDict):
    id: ReadOnly[str]
    custom_id: ReadOnly[str]
    response: ReadOnly[_ResultResponse]
    error: ReadOnly[None]


class BatchInputLine(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    custom_id: str
    method: Literal["POST"]
    url: str
    body: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class InvalidBatchInput:
    line_number: int | None
    reason: str

    def describe(self) -> str:
        return f"line {self.line_number}: {self.reason}" if self.line_number is not None else self.reason


@dataclass(frozen=True, slots=True)
class RowOutcome:
    custom_id: str
    status_code: int
    body: Mapping[str, object]
    succeeded: bool


@dataclass(frozen=True, slots=True)
class _BatchRun:
    unified_batch_id: str
    llm_batch_id: str
    model: str
    endpoint: BatchEndpoint
    lines: tuple[BatchInputLine, ...]
    user_api_key_dict: UserAPIKeyAuth
    request_tags: tuple[str, ...]


@runtime_checkable
class ManagedBatchStore(Protocol):
    def get_unified_batch_id(self, batch_id: str, model_id: str) -> str: ...

    async def get_unified_file_id(
        self, file_id: str, litellm_parent_otel_span: object | None = None
    ) -> LiteLLM_ManagedFileTable | None: ...

    async def store_unified_object_id(
        self,
        unified_object_id: str,
        file_object: LiteLLMBatch,
        litellm_parent_otel_span: object | None,
        model_object_id: str,
        file_purpose: Literal["batch", "fine-tune", "response"],
        user_api_key_dict: UserAPIKeyAuth,
        request_tags: Sequence[str] | None = None,
        persist_attribution: bool = False,
        create_if_missing: bool = True,
        batch_processed: bool = False,
    ) -> None: ...


class _StorageBackendFactory(Protocol):
    def __call__(self, backend_type: str, prisma_client: PrismaClient | None = None) -> BaseFileStorageBackend: ...


class _ResultFileUploader(Protocol):
    def __call__(
        self,
        file_data: Mapping[str, object],
        target_storage: str,
        target_model_names: Sequence[str],
        purpose: OpenAIFilesPurpose,
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
        prisma_client: PrismaClient | None = None,
    ) -> Awaitable[OpenAIFileObject]: ...


@runtime_checkable
class _RouterCall(Protocol):
    def __call__(self, **params: object) -> Awaitable[object]: ...  # kwargs-ok: the request body is passed as keywords


def litellm_executed_provider_of(credentials: Mapping[str, object]) -> str | None:
    explicit_provider: Final = credentials.get("custom_llm_provider")
    provider: Final = (
        explicit_provider if isinstance(explicit_provider, str) else _provider_of(credentials.get("model"))
    )
    return provider if provider in LITELLM_EXECUTED_BATCH_PROVIDERS else None


class _HttpGetter(Protocol):
    async def get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float | httpx.Timeout | None = None
    ) -> httpx.Response: ...


class FilesApiProbe(Protocol):
    async def __call__(self, api_base: str, api_key: str | None) -> bool: ...


class BodyRejection(Protocol):
    def __call__(self, body: Mapping[str, object], /) -> str | None: ...


async def upstream_lacks_files_api(api_base: str, api_key: str | None, http_client: _HttpGetter | None = None) -> bool:
    client: Final = http_client or get_async_httpx_client(llm_provider=LlmProviders.HOSTED_VLLM)
    try:
        response: Final = await client.get(
            f"{api_base.rstrip('/')}/files",
            headers=(
                {"Authorization": f"Bearer {api_key}"}  # mutable-ok: AsyncHTTPHandler.get wants a plain dict
                if api_key
                else None
            ),
            timeout=_FILES_API_PROBE_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == httpx.codes.NOT_FOUND


def _upstream_of(credentials: Mapping[str, object], provider: str) -> tuple[str, str | None] | None:
    model: Final = credentials.get("model")
    api_base: Final = credentials.get("api_base")
    api_key: Final = credentials.get("api_key")
    if not isinstance(model, str):
        return None
    try:
        _, _, resolved_api_key, resolved_api_base = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=provider,
            api_base=api_base if isinstance(api_base, str) else None,
            api_key=api_key if isinstance(api_key, str) else None,
        )
    except Exception:  # noqa: BLE001  # get_llm_provider raises on a model it cannot map, which means nothing to probe
        return None
    return None if resolved_api_base is None else (resolved_api_base, resolved_api_key)


async def litellm_executed_provider_for(
    credentials: Mapping[str, object], lacks_files_api: FilesApiProbe = upstream_lacks_files_api
) -> str | None:
    provider: Final = litellm_executed_provider_of(credentials)
    if provider is None:
        return None
    upstream: Final = _upstream_of(credentials, provider)
    if upstream is None:
        return None
    return provider if await lacks_files_api(*upstream) else None


async def resolve_litellm_executed_provider(
    llm_router: "Router",
    model: str,
    team_id: str | None,
    lacks_files_api: FilesApiProbe = upstream_lacks_files_api,
) -> str | None:
    credentials: Final = llm_router.get_deployment_credentials_with_provider(model_id=model, team_id=team_id)
    return None if credentials is None else await litellm_executed_provider_for(credentials, lacks_files_api)


def _provider_of(model: object) -> str | None:
    if not isinstance(model, str):
        return None
    try:
        return litellm.get_llm_provider(model=model)[1]
    except Exception:  # noqa: BLE001  # get_llm_provider raises on an unknown model, which means no provider
        return None


def _validation_reason(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" if item["loc"] else item["msg"]
        for item in error.errors()
    )


def _accept_every_body(_body: Mapping[str, object]) -> str | None:
    return None


def _parse_line(
    line_number: int, raw: bytes, endpoint: BatchEndpoint, reject_body: BodyRejection
) -> BatchInputLine | InvalidBatchInput:
    try:
        line: Final = BatchInputLine.model_validate_json(raw)
    except ValidationError as e:
        return InvalidBatchInput(line_number, _validation_reason(e))
    if line.url != endpoint:
        return InvalidBatchInput(line_number, f"url {line.url!r} does not match the batch endpoint {endpoint!r}")
    if line.body.get("stream"):
        return InvalidBatchInput(line_number, "streaming requests are not supported in a batch")
    rejection: Final = reject_body(line.body)
    if rejection is not None:
        return InvalidBatchInput(line_number, rejection)
    return line


def parse_batch_input(
    content: bytes, endpoint: BatchEndpoint, reject_body: BodyRejection = _accept_every_body
) -> tuple[BatchInputLine, ...] | InvalidBatchInput:
    raw_lines: Final = tuple((number, raw) for number, raw in enumerate(content.splitlines(), start=1) if raw.strip())
    if not raw_lines:
        return InvalidBatchInput(None, "the input file has no requests")
    parsed: Final = tuple(_parse_line(number, raw, endpoint, reject_body) for number, raw in raw_lines)
    first_invalid: Final = next((item for item in parsed if isinstance(item, InvalidBatchInput)), None)
    if first_invalid is not None:
        return first_invalid
    lines: Final = tuple(item for item in parsed if isinstance(item, BatchInputLine))
    custom_ids: Final = sorted(line.custom_id for line in lines)
    duplicate: Final = next((first for first, second in pairwise(custom_ids) if first == second), None)
    if duplicate is not None:
        return InvalidBatchInput(None, f"custom_id {duplicate!r} is used more than once")
    return lines


def batch_error(status_code: int, message: str) -> ProxyException:
    error_type: Final = "invalid_request_error" if status_code < 500 else ProxyErrorTypes.internal_server_error.value
    return ProxyException(message=message, type=error_type, param=None, code=status_code)


def _validate_endpoint(endpoint: object) -> BatchEndpoint:
    try:
        return _BATCH_ENDPOINT_ADAPTER.validate_python(endpoint)
    except ValidationError:
        raise batch_error(400, f"endpoint {endpoint!r} is not supported for a LiteLLM-executed batch")


def _status_code_of(error: Exception) -> int:
    status_code: Final[object] = getattr(error, "status_code", None)
    return status_code if isinstance(status_code, int) else 500


def _batch_of(blob: object) -> LiteLLMBatch:
    return LiteLLMBatch.model_validate_json(blob) if isinstance(blob, str) else LiteLLMBatch.model_validate(blob)


def _error_body(error: Exception) -> _ErrorBody:
    body: Final[_ErrorBody] = {
        "error": {"message": str(error), "type": type(error).__name__, "param": None, "code": None}
    }
    return body


def _result_line(outcome: RowOutcome) -> _ResultLine:
    line: Final[_ResultLine] = {
        "id": f"batch_req_{uuid_module.uuid4().hex[:24]}",
        "custom_id": outcome.custom_id,
        "response": {
            "status_code": outcome.status_code,
            "request_id": f"req_{uuid_module.uuid4().hex[:24]}",
            "body": outcome.body,
        },
        "error": None,
    }
    return line


def _dump(response: object) -> Mapping[str, object]:
    if isinstance(response, BaseModel):
        return response.model_dump(mode="json")
    raise TypeError(f"Batch rows must return a single response object, got {type(response).__name__}")


def _resolve_transition(current_status: str, requested: BatchStatus) -> BatchStatus:
    if current_status != "cancelling":
        return requested
    return _CANCELLING_TRANSITIONS.get(requested, requested)


def executed_batch_runner_lost(status: str, updated_at: datetime) -> bool:
    if status in TERMINAL_BATCH_STATUSES:
        return False
    return (datetime.now(timezone.utc) - updated_at).total_seconds() > _STALE_AFTER_SECONDS


def _llm_batch_id_of(unified_batch_id: str) -> str:
    return get_batch_id_from_unified_batch_id(convert_b64_uid_to_unified_uid(unified_batch_id))


class _StopWatch:
    def __init__(self, load_status: Callable[[], Awaitable[str | None]], interval_seconds: float) -> None:
        self._load_status = load_status
        self._interval_seconds = interval_seconds
        self._checked_at = float("-inf")
        self._stopped = False

    async def stopped(self) -> bool:
        if self._stopped:
            return True
        now: Final = time.monotonic()
        if now - self._checked_at < self._interval_seconds:
            return False
        self._checked_at = now
        self._stopped = await self._load_status() in _STOP_STATUSES
        return self._stopped


class LiteLLMExecutedBatchRunner:
    def __init__(
        self,
        llm_router: "Router",
        prisma_client: PrismaClient,
        managed_files: ManagedBatchStore,
        proxy_logging_obj: ProxyLogging,
        general_settings: Mapping[str, object],
        concurrency: int = LITELLM_EXECUTED_BATCH_CONCURRENCY,
        heartbeat_seconds: float = _HEARTBEAT_SECONDS,
        storage_backend_factory: _StorageBackendFactory = get_storage_backend,
        upload_result_file: _ResultFileUploader = StorageBackendFileService.upload_file_to_storage_backend,
    ) -> None:
        self.llm_router = llm_router
        self.prisma_client = prisma_client
        self.managed_files = managed_files
        self.proxy_logging_obj = proxy_logging_obj
        self.general_settings = general_settings
        self.concurrency = concurrency
        self.heartbeat_seconds = heartbeat_seconds
        self.storage_backend_factory = storage_backend_factory
        self.upload_result_file = upload_result_file

    async def create(
        self,
        create_request: LiteLLMBatchCreateRequest,
        unified_input_file_id: str,
        model: str,
        provider: str,
        user_api_key_dict: UserAPIKeyAuth,
        request_tags: Sequence[str] | None,
    ) -> LiteLLMBatch:
        endpoint: Final = _validate_endpoint(create_request.get("endpoint"))
        content: Final = await self._download_input(unified_input_file_id, user_api_key_dict)
        parsed: Final = parse_batch_input(content, endpoint, self._body_rejection(model))
        if isinstance(parsed, InvalidBatchInput):
            raise batch_error(400, f"Invalid batch input file: {parsed.describe()}")
        llm_batch_id: Final = f"{LITELLM_EXECUTED_BATCH_ID_PREFIX}{uuid_module.uuid4().hex}"
        model_id: Final = next(iter(self.llm_router.get_model_ids(model_name=model)), model)
        unified_batch_id: Final = self.managed_files.get_unified_batch_id(batch_id=llm_batch_id, model_id=model_id)
        created_at: Final = int(time.time())
        batch: Final = LiteLLMBatch(
            id=unified_batch_id,
            object="batch",
            endpoint=endpoint,
            input_file_id=unified_input_file_id,
            completion_window="24h",
            status="validating",
            created_at=created_at,
            expires_at=created_at + _COMPLETION_WINDOW_SECONDS,
            metadata=create_request.get("metadata"),
            model=model,
            request_counts=BatchRequestCounts(completed=0, failed=0, total=len(parsed)),
        )
        await self.managed_files.store_unified_object_id(
            unified_object_id=unified_batch_id,
            file_object=batch,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            model_object_id=llm_batch_id,
            file_purpose="batch",
            user_api_key_dict=user_api_key_dict,
            request_tags=request_tags,
            persist_attribution=True,
            batch_processed=True,
        )
        _record_batch_created(model, provider, user_api_key_dict)
        run: Final = _BatchRun(
            unified_batch_id=unified_batch_id,
            llm_batch_id=llm_batch_id,
            model=model,
            endpoint=endpoint,
            lines=parsed,
            user_api_key_dict=user_api_key_dict,
            request_tags=tuple(request_tags or ()),
        )
        task: Final = asyncio.create_task(self._run(run))
        _RUNNING_BATCHES.add(task)
        task.add_done_callback(_RUNNING_BATCHES.discard)
        return batch

    async def cancel(self, unified_batch_id: str, user_api_key_dict: UserAPIKeyAuth) -> LiteLLMBatch:
        current: Final = await self._load_batch(unified_batch_id)
        if current is None:
            raise batch_error(404, f"Batch {unified_batch_id} not found")
        if current.status in TERMINAL_BATCH_STATUSES:
            raise batch_error(400, f"Cannot cancel a batch with status '{current.status}'")
        if current.status == "cancelling":
            return current
        cancelling: Final = current.model_copy(
            update=MappingProxyType({"status": "cancelling", "cancelling_at": int(time.time())})
        )
        await self._store(cancelling, user_api_key_dict)
        return cancelling

    async def fail_abandoned(self, batch: LiteLLMBatch, user_api_key_dict: UserAPIKeyAuth) -> LiteLLMBatch:
        error: Final = BatchError(message=_RUNNER_LOST_MESSAGE, code="runner_lost")
        errors: Final = Errors(data=[error], object="list")  # mutable-ok: Errors.data is typed as a list
        failed: Final = batch.model_copy(
            update=MappingProxyType({"status": "failed", "failed_at": int(time.time()), "errors": errors})
        )
        await self._store(failed, user_api_key_dict)
        return failed

    def _body_rejection(self, model: str) -> BodyRejection:
        def reject(body: Mapping[str, object]) -> str | None:
            try:
                is_request_body_safe(
                    request_body=dict(body),  # mutable-ok: is_request_body_safe takes a dict
                    general_settings=dict(self.general_settings),  # mutable-ok: is_request_body_safe takes a dict
                    llm_router=self.llm_router,
                    model=model,
                )
            except ValueError as e:
                return str(e)
            return None

        return reject

    async def _download_input(self, unified_input_file_id: str, user_api_key_dict: UserAPIKeyAuth) -> bytes:
        stored: Final = await self.managed_files.get_unified_file_id(
            unified_input_file_id, litellm_parent_otel_span=user_api_key_dict.parent_otel_span
        )
        if stored is None or not stored.storage_backend or not stored.storage_url:
            raise batch_error(
                400,
                f"LiteLLM does not hold the content of input file {unified_input_file_id}: "
                f"{LITELLM_EXECUTED_BATCH_UPLOAD_GUIDANCE}",
            )
        try:
            backend: Final = self.storage_backend_factory(stored.storage_backend, prisma_client=self.prisma_client)
            return await backend.download_file(stored.storage_url)
        except ValueError as e:
            raise batch_error(400, str(e))

    async def _run(self, run: _BatchRun) -> None:
        heartbeat: Final = asyncio.create_task(self._heartbeat(run))
        try:
            await self._execute(run)
        except Exception as e:  # noqa: BLE001  # whatever fails, the batch must end up marked failed
            verbose_proxy_logger.exception("LiteLLM-executed batch %s failed: %s", run.unified_batch_id, e)
            error: Final = BatchError(message=str(e), code="internal_error")
            errors: Final = Errors(data=[error], object="list")  # mutable-ok: Errors.data is typed as a list
            try:
                await self._advance(run, "failed", MappingProxyType({"errors": errors}))
            except Exception as advance_error:  # noqa: BLE001  # a failed status write is logged, never raised
                verbose_proxy_logger.exception(
                    "LiteLLM-executed batch %s could not be marked failed: %s", run.unified_batch_id, advance_error
                )
        finally:
            heartbeat.cancel()

    async def _heartbeat(self, run: _BatchRun) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await self._touch(run)
            except Exception as e:  # noqa: BLE001  # a missed beat is logged and the next one retries
                verbose_proxy_logger.warning("LiteLLM-executed batch %s heartbeat failed: %s", run.unified_batch_id, e)

    async def _touch(self, run: _BatchRun) -> None:
        await ManagedObjectRepository(self.prisma_client).table.update_many(
            where={"unified_object_id": run.unified_batch_id},  # mutable-ok: Prisma filter
            data={"updated_by": run.user_api_key_dict.user_id},  # mutable-ok: Prisma payload
        )

    async def _execute(self, run: _BatchRun) -> None:
        await self._advance(run, "in_progress")
        watch: Final = _StopWatch(lambda: self._load_status(run.unified_batch_id), _CANCEL_POLL_SECONDS)
        semaphore: Final = asyncio.Semaphore(self.concurrency)
        results: Final = await asyncio.gather(*(self._run_row(run, line, watch, semaphore) for line in run.lines))
        outcomes: Final = tuple(outcome for outcome in results if outcome is not None)
        if await self._advance(run, "finalizing") is None:
            return
        succeeded: Final = tuple(outcome for outcome in outcomes if outcome.succeeded)
        failed: Final = tuple(outcome for outcome in outcomes if not outcome.succeeded)
        output_file_id: Final = await self._upload_results(run, "output", succeeded)
        error_file_id: Final = await self._upload_results(run, "error", failed)
        request_counts: Final = BatchRequestCounts(completed=len(succeeded), failed=len(failed), total=len(run.lines))
        await self._advance(
            run,
            "completed",
            MappingProxyType(
                {"output_file_id": output_file_id, "error_file_id": error_file_id, "request_counts": request_counts}
            ),
        )

    async def _run_row(
        self, run: _BatchRun, line: BatchInputLine, watch: _StopWatch, semaphore: asyncio.Semaphore
    ) -> RowOutcome | None:
        async with semaphore:
            if await watch.stopped():
                return None
            try:
                body: Final = await self._dispatch(run, line)
            except Exception as e:  # noqa: BLE001  # a provider error becomes the row's error line, never a crashed batch
                return RowOutcome(
                    custom_id=line.custom_id, status_code=_status_code_of(e), body=_error_body(e), succeeded=False
                )
            return RowOutcome(custom_id=line.custom_id, status_code=200, body=body, succeeded=True)

    async def _dispatch(self, run: _BatchRun, line: BatchInputLine) -> Mapping[str, object]:
        params: Final = MappingProxyType({**line.body, "model": run.model, "metadata": self._row_metadata(run)})
        return _dump(await self._router_call(run.endpoint)(**params))

    def _router_call(self, endpoint: BatchEndpoint) -> _RouterCall:
        method: Final[object] = getattr(self.llm_router, _ROUTER_METHODS[endpoint], None)
        if not isinstance(method, _RouterCall):
            raise TypeError(f"the router has no callable for {endpoint}")
        return method

    def _row_metadata(self, run: _BatchRun) -> dict[str, object]:  # mutable-ok: router updates metadata in place
        return {  # mutable-ok: the router updates request metadata in place
            **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(run.user_api_key_dict),
            "user_api_key": run.user_api_key_dict.api_key,
            "user_api_end_user_max_budget": run.user_api_key_dict.end_user_max_budget,
            "tags": list(run.request_tags),  # mutable-ok: litellm types request tags as a list
            "batch_id": run.unified_batch_id,
        }

    async def _upload_results(
        self, run: _BatchRun, kind: Literal["output", "error"], outcomes: Sequence[RowOutcome]
    ) -> str | None:
        if not outcomes:
            return None
        content: Final = "".join(f"{json.dumps(_result_line(outcome))}\n" for outcome in outcomes).encode()
        file_data: Final[ExtractedFileData] = {
            "filename": f"{run.llm_batch_id}_{kind}.jsonl",
            "content": content,
            "content_type": "application/jsonl",
            "headers": _NO_HEADERS,
        }
        file_object: Final = await self.upload_result_file(
            file_data=file_data,
            target_storage=LITELLM_DB_STORAGE_BACKEND_NAME,
            target_model_names=(run.model,),
            purpose="batch_output",
            proxy_logging_obj=self.proxy_logging_obj,
            user_api_key_dict=run.user_api_key_dict,
            prisma_client=self.prisma_client,
        )
        return file_object.id

    async def _advance(
        self, run: _BatchRun, requested: BatchStatus, fields: Mapping[str, object] = _NO_FIELDS
    ) -> BatchStatus | None:
        current: Final = await self._load_batch(run.unified_batch_id)
        if current is None:
            raise RuntimeError(f"Batch {run.unified_batch_id} is no longer stored")
        if current.status in TERMINAL_BATCH_STATUSES:
            return None
        status: Final = _resolve_transition(current.status, requested)
        updated: Final = current.model_copy(
            update=MappingProxyType({**fields, "status": status, f"{status}_at": int(time.time())})
        )
        await self._store(updated, run.user_api_key_dict)
        return status

    async def _store(self, batch: LiteLLMBatch, user_api_key_dict: UserAPIKeyAuth) -> None:
        await self.managed_files.store_unified_object_id(
            unified_object_id=batch.id,
            file_object=batch,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            model_object_id=_llm_batch_id_of(batch.id),
            file_purpose="batch",
            user_api_key_dict=user_api_key_dict,
            create_if_missing=False,
        )

    async def _find_row(self, unified_batch_id: str) -> "prisma_models.LiteLLM_ManagedObjectTable | None":
        return await ManagedObjectRepository(self.prisma_client).table.find_first(
            where={"unified_object_id": unified_batch_id}  # mutable-ok: Prisma filter
        )

    async def _load_batch(self, unified_batch_id: str) -> LiteLLMBatch | None:
        row: Final = await self._find_row(unified_batch_id)
        return None if row is None or not row.file_object else _batch_of(row.file_object)

    async def _load_status(self, unified_batch_id: str) -> str | None:
        row: Final = await self._find_row(unified_batch_id)
        return row.status if row is not None else None


def _record_batch_created(model: str, provider: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    prometheus_logger: Final = PrometheusLogger.get_instance()
    if prometheus_logger is None:
        return
    prometheus_logger.record_managed_batch_created(
        model=model,
        api_provider=provider,
        user=user_api_key_dict.user_id or "",
        user_email=user_api_key_dict.user_email or "",
        api_key_alias=user_api_key_dict.key_alias or "",
    )
