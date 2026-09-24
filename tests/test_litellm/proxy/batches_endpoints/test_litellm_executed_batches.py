import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles
from openai.types.batch_request_counts import BatchRequestCounts

from litellm.models.managed_files import LiteLLM_ManagedFileTable
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.batches_endpoints import litellm_executed_batches
from litellm.proxy.batches_endpoints.litellm_executed_batches import (
    BatchEndpoint,
    BatchInputLine,
    BatchStatus,
    InvalidBatchInput,
    LiteLLMExecutedBatchRunner,
    _resolve_transition,
    executed_batch_runner_lost,
    litellm_executed_provider_for,
    litellm_executed_provider_of,
    litellm_stored_batch_input_provider_of,
    parse_batch_input,
    resolve_litellm_executed_provider,
    upstream_lacks_files_api,
)
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
    get_batch_id_from_unified_batch_id,
    is_litellm_executed_batch,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.managed_batch_repository import ManagedBatchRepository
from litellm.router import Router
from litellm.types.llms.openai import LiteLLMBatchCreateRequest, OpenAIFileObject, OpenAIFilesPurpose
from litellm.types.utils import EmbeddingResponse, LiteLLMBatch, ModelResponse, SpecialEnums

BATCH_MODEL: Final = "batch-model"
DEPLOYMENT_ID: Final = "deployment-id-1"
INPUT_FILE_ID: Final = "unified-input-file"
STORAGE_BACKEND: Final = "s3"
STORAGE_URL: Final = "s3://bucket/input.jsonl"
CHAT_ENDPOINT: Final = "/v1/chat/completions"
ROUTER_METHODS: Final = ("acompletion", "atext_completion", "aembedding", "aresponses")
ALL_STATUSES: Final[tuple[BatchStatus, ...]] = (
    "in_progress",
    "finalizing",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
    "expired",
)


def chat_row(custom_id: str, content: str, **body_extra: object) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_ENDPOINT,
        "body": {"model": "row-model", "messages": [{"role": "user", "content": content}], **body_extra},
    }


def jsonl(*rows: Mapping[str, object]) -> bytes:
    return "".join(f"{json.dumps(row)}\n" for row in rows).encode()


TWO_CHAT_ROWS: Final = jsonl(chat_row("row-1", "hi 1"), chat_row("row-2", "hi 2"))


def chat_response(content: str) -> ModelResponse:
    return ModelResponse(
        id=f"chatcmpl-{content}",
        model=BATCH_MODEL,
        choices=[{"index": 0, "message": {"role": "assistant", "content": f"echo {content}"}, "finish_reason": "stop"}],
    )


def managed_input_file(storage_backend: str | None = STORAGE_BACKEND) -> LiteLLM_ManagedFileTable:
    return LiteLLM_ManagedFileTable(
        unified_file_id=INPUT_FILE_ID,
        model_mappings={},
        flat_model_file_ids=[],
        storage_backend=storage_backend,
        storage_url=STORAGE_URL,
    )


def batch_request(endpoint: str) -> LiteLLMBatchCreateRequest:
    return cast(
        "LiteLLMBatchCreateRequest",
        {"endpoint": endpoint, "input_file_id": INPUT_FILE_ID, "completion_window": "24h"},
    )


class ProviderRateLimited(Exception):
    status_code = 429


@dataclass(frozen=True, slots=True)
class StoredObject:
    file_object: str
    status: str
    updated_at: datetime

    def batch(self) -> LiteLLMBatch:
        return LiteLLMBatch.model_validate_json(self.file_object)


@dataclass(frozen=True, slots=True)
class StoreCall:
    unified_object_id: str
    model_object_id: str
    status: str
    request_tags: tuple[str, ...] | None
    persist_attribution: bool
    batch_processed: bool


@dataclass(frozen=True, slots=True)
class StatusWrite:
    unified_object_id: str
    status: str
    columns: frozenset[str]


STATUS_WRITE_COLUMNS: Final = frozenset({"file_object", "status", "updated_by"})
STALE: Final = timedelta(seconds=litellm_executed_batches._STALE_AFTER_SECONDS + 20)


class FakeManagedBatchStore:
    def __init__(self, files: Mapping[str, LiteLLM_ManagedFileTable]) -> None:
        self.files = files
        self.objects: dict[str, StoredObject] = {}
        self.calls: list[StoreCall] = []

    def get_unified_batch_id(self, batch_id: str, model_id: str) -> str:
        return SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(model_id, batch_id)

    async def get_unified_file_id(
        self, file_id: str, litellm_parent_otel_span: object | None = None
    ) -> LiteLLM_ManagedFileTable | None:
        return self.files.get(file_id)

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
        batch_processed: bool = False,
    ) -> None:
        self.calls.append(
            StoreCall(
                unified_object_id=unified_object_id,
                model_object_id=model_object_id,
                status=file_object.status,
                request_tags=tuple(request_tags) if request_tags is not None else None,
                persist_attribution=persist_attribution,
                batch_processed=batch_processed,
            )
        )
        self.write(file_object)

    def write(self, batch: LiteLLMBatch, age: timedelta = timedelta(0)) -> None:
        self.objects[batch.id] = StoredObject(
            file_object=batch.model_dump_json(), status=batch.status, updated_at=datetime.now(timezone.utc) - age
        )

    def batch(self, unified_batch_id: str) -> LiteLLMBatch:
        return self.objects[unified_batch_id].batch()


REAL_HOOK: Final = _PROXY_LiteLLMManagedFiles(internal_usage_cache=MagicMock(), prisma_client=MagicMock())


class RealIdManagedBatchStore(FakeManagedBatchStore):
    def get_unified_batch_id(self, batch_id: str, model_id: str) -> str:
        return REAL_HOOK.get_unified_batch_id(batch_id=batch_id, model_id=model_id)


def row_matches(row: StoredObject, where: Mapping[str, object]) -> bool:
    if "status" in where and row.status != where["status"]:
        return False
    match where.get("updated_at"):
        case {"lt": datetime() as before}:
            return row.updated_at < before
        case _:
            return True


class FakeManagedObjectTable:
    def __init__(self, objects: dict[str, StoredObject]) -> None:
        self.objects = objects
        self.touches: list[tuple[str, str | None]] = []
        self.writes: list[StatusWrite] = []
        self.after_read: Callable[[StoredObject | None], None] | None = None

    async def find_first(self, where: Mapping[str, str]) -> StoredObject | None:
        row = self.objects.get(where["unified_object_id"])
        if self.after_read is not None:
            self.after_read(row)
        return row

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, str | None]) -> int:
        unified_object_id = str(where["unified_object_id"])
        row = self.objects.get(unified_object_id)
        if row is None or not row_matches(row, where):
            return 0
        now = datetime.now(timezone.utc)
        if "status" not in data:
            self.touches.append((unified_object_id, data["updated_by"]))
            self.objects[unified_object_id] = StoredObject(row.file_object, row.status, now)
            return 1
        self.writes.append(StatusWrite(unified_object_id, str(data["status"]), frozenset(data)))
        self.objects[unified_object_id] = StoredObject(str(data["file_object"]), str(data["status"]), now)
        return 1


class FakeDb:
    def __init__(self, objects: dict[str, StoredObject]) -> None:
        self.litellm_managedobjecttable = FakeManagedObjectTable(objects)


class FakePrismaClient:
    def __init__(self, objects: dict[str, StoredObject]) -> None:
        self.db = FakeDb(objects)


class FakeRouter:
    def __init__(self) -> None:
        self.acompletion = AsyncMock(return_value=chat_response("default"))
        self.atext_completion = AsyncMock(return_value=chat_response("default"))
        self.aembedding = AsyncMock(
            return_value=EmbeddingResponse(
                model=BATCH_MODEL, data=[{"embedding": [0.1], "index": 0, "object": "embedding"}]
            )
        )
        self.aresponses = AsyncMock(return_value=chat_response("default"))

    def get_model_ids(self, model_name: str) -> list[str]:
        return [DEPLOYMENT_ID] if model_name == BATCH_MODEL else []

    def get_model_group_info(self, model_group: str) -> None:
        return None


class FakeStorageBackend:
    def __init__(self, contents: Mapping[str, bytes]) -> None:
        self.contents = contents
        self.downloads: list[str] = []

    async def download_file(self, storage_url: str) -> bytes:
        self.downloads.append(storage_url)
        return self.contents[storage_url]


class FakeStorageBackendFactory:
    def __init__(self, backend: FakeStorageBackend, error: ValueError | None) -> None:
        self.backend = backend
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def __call__(self, backend_type: str, prisma_client: object = None) -> FakeStorageBackend:
        self.calls.append((backend_type, prisma_client))
        if self.error is not None:
            raise self.error
        return self.backend


@dataclass(frozen=True, slots=True)
class UploadCall:
    content: bytes
    filename: str
    target_storage: str
    target_model_names: tuple[str, ...]
    purpose: str
    user_api_key_dict: UserAPIKeyAuth
    prisma_client: object

    def lines(self) -> dict[str, dict[str, object]]:
        parsed = tuple(json.loads(line) for line in self.content.decode().splitlines())
        return {str(line["custom_id"]): line for line in parsed}


class FakeResultFileUploader:
    def __init__(self, error: Exception | None) -> None:
        self.error = error
        self.calls: list[UploadCall] = []

    async def __call__(
        self,
        file_data: Mapping[str, object],
        target_storage: str,
        target_model_names: list[str],
        purpose: OpenAIFilesPurpose,
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
        prisma_client: object = None,
    ) -> OpenAIFileObject:
        content = file_data["content"]
        assert isinstance(content, bytes)
        self.calls.append(
            UploadCall(
                content=content,
                filename=str(file_data["filename"]),
                target_storage=target_storage,
                target_model_names=tuple(target_model_names),
                purpose=purpose,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
            )
        )
        if self.error is not None:
            raise self.error
        return OpenAIFileObject(
            id=f"unified-output-{len(self.calls)}",
            object="file",
            bytes=len(content),
            created_at=0,
            filename=str(file_data["filename"]),
            purpose=purpose,
            status="uploaded",
        )


@dataclass(frozen=True, slots=True)
class Harness:
    runner: LiteLLMExecutedBatchRunner
    store: FakeManagedBatchStore
    router: FakeRouter
    uploads: FakeResultFileUploader
    storage: FakeStorageBackend
    storage_factory: FakeStorageBackendFactory
    prisma: FakePrismaClient
    user: UserAPIKeyAuth

    async def create(self, endpoint: str = CHAT_ENDPOINT) -> LiteLLMBatch:
        return await self.runner.create(
            create_request=batch_request(endpoint),
            unified_input_file_id=INPUT_FILE_ID,
            model=BATCH_MODEL,
            provider="hosted_vllm",
            user_api_key_dict=self.user,
            request_tags=["tag-a"],
        )

    async def create_and_finish(self, endpoint: str = CHAT_ENDPOINT) -> tuple[LiteLLMBatch, LiteLLMBatch]:
        created = await self.create(endpoint)
        await asyncio.gather(*list(litellm_executed_batches._RUNNING_BATCHES))
        return created, self.store.batch(created.id)

    @property
    def table(self) -> FakeManagedObjectTable:
        return self.prisma.db.litellm_managedobjecttable

    def written_statuses(self) -> list[str]:
        return [write.status for write in self.table.writes]


def make_runner(
    content: bytes = TWO_CHAT_ROWS,
    concurrency: int = 4,
    files: Mapping[str, LiteLLM_ManagedFileTable] | None = None,
    upload_error: Exception | None = None,
    storage_error: ValueError | None = None,
    store_factory: Callable[[Mapping[str, LiteLLM_ManagedFileTable]], FakeManagedBatchStore] = FakeManagedBatchStore,
    general_settings: Mapping[str, object] = MappingProxyType({}),
    heartbeat_seconds: float = 30.0,
    completion_window_seconds: float = 24 * 60 * 60,
) -> Harness:
    store = store_factory({INPUT_FILE_ID: managed_input_file()} if files is None else files)
    router = FakeRouter()
    uploads = FakeResultFileUploader(upload_error)
    storage = FakeStorageBackend({STORAGE_URL: content})
    storage_factory = FakeStorageBackendFactory(storage, storage_error)
    prisma = FakePrismaClient(store.objects)
    user = UserAPIKeyAuth(
        api_key="sk-batch-key", user_id="user-1", team_id="team-1", key_alias="alias-1", user_email="user@example.com"
    )
    runner = LiteLLMExecutedBatchRunner(
        llm_router=cast("Router", router),
        prisma_client=cast("PrismaClient", prisma),
        managed_files=store,
        batches=ManagedBatchRepository(prisma),
        proxy_logging_obj=MagicMock(spec=ProxyLogging),
        general_settings=general_settings,
        concurrency=concurrency,
        heartbeat_seconds=heartbeat_seconds,
        completion_window_seconds=completion_window_seconds,
        storage_backend_factory=storage_factory,
        upload_result_file=uploads,
    )
    return Harness(runner, store, router, uploads, storage, storage_factory, prisma, user)


def seeded_batch(
    store: FakeManagedBatchStore, status: Literal["in_progress", "completed"], age: timedelta = timedelta(0)
) -> LiteLLMBatch:
    batch = LiteLLMBatch(
        id=store.get_unified_batch_id(batch_id="litellm_batch_seed", model_id=DEPLOYMENT_ID),
        object="batch",
        endpoint=CHAT_ENDPOINT,
        input_file_id=INPUT_FILE_ID,
        completion_window="24h",
        status=status,
        created_at=1,
        model=BATCH_MODEL,
    )
    store.write(batch, age)
    return batch


@pytest.mark.parametrize(
    ("content", "line_number", "reason_fragment"),
    [
        (b"", None, "no requests"),
        (b"\n   \n", None, "no requests"),
        (b"{not json", 1, "JSON"),
        (jsonl({"custom_id": "a", "method": "POST", "url": CHAT_ENDPOINT}), 1, "body"),
        (jsonl({**chat_row("a", "hi"), "extra_field": 1}), 1, "extra_field"),
        (
            jsonl(chat_row("a", "hi")) + b"\n" + jsonl({**chat_row("b", "hi"), "url": "/v1/embeddings"}),
            3,
            "/v1/embeddings",
        ),
        (jsonl(chat_row("a", "hi", stream=True)), 1, "streaming"),
        (jsonl(chat_row("a", "hi"), chat_row("a", "again")), None, "'a'"),
    ],
    ids=["empty", "blank lines", "not json", "missing body", "unknown field", "url mismatch", "stream", "duplicate id"],
)
def test_parse_batch_input_rejects(content: bytes, line_number: int | None, reason_fragment: str) -> None:
    result = parse_batch_input(content, CHAT_ENDPOINT)
    assert isinstance(result, InvalidBatchInput)
    assert result.line_number == line_number
    assert reason_fragment in result.reason


def test_parse_batch_input_keeps_every_request_and_skips_blank_lines() -> None:
    content = b"\n" + jsonl(chat_row("a", "hi 1")) + b"\n" + jsonl(chat_row("b", "hi 2")) + b"\n\n"
    lines = parse_batch_input(content, CHAT_ENDPOINT)
    assert isinstance(lines, tuple)
    assert [line.custom_id for line in lines] == ["a", "b"]
    assert lines[1] == BatchInputLine(
        custom_id="b",
        method="POST",
        url=CHAT_ENDPOINT,
        body={"model": "row-model", "messages": [{"role": "user", "content": "hi 2"}]},
    )


@pytest.mark.parametrize("current", ["validating", "in_progress", "finalizing"])
@pytest.mark.parametrize("requested", ALL_STATUSES)
def test_resolve_transition_keeps_the_requested_status_unless_cancelling(current: str, requested: BatchStatus) -> None:
    assert _resolve_transition(current, requested) == requested


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("completed", "cancelled"),
        ("expired", "cancelled"),
        ("in_progress", "cancelling"),
        ("finalizing", "cancelling"),
        ("failed", "failed"),
        ("cancelling", "cancelling"),
        ("cancelled", "cancelled"),
    ],
)
def test_resolve_transition_from_cancelling(requested: BatchStatus, expected: BatchStatus) -> None:
    assert _resolve_transition("cancelling", requested) == expected


@pytest.mark.parametrize(
    ("status", "age_seconds", "lost"),
    [
        ("validating", 200, True),
        ("in_progress", 200, True),
        ("in_progress", 100, False),
        ("finalizing", 200, True),
        ("cancelling", 200, True),
        ("completed", 200, False),
        ("failed", 200, False),
        ("cancelled", 200, False),
        ("expired", 200, False),
    ],
)
def test_executed_batch_runner_lost_only_for_a_stale_non_terminal_batch(
    status: str, age_seconds: int, lost: bool
) -> None:
    updated_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    assert executed_batch_runner_lost(status, updated_at) is lost


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"custom_llm_provider": "hosted_vllm", "model": "openai/gpt-4o"}, "hosted_vllm"),
        ({"model": "hosted_vllm/qwen"}, "hosted_vllm"),
        ({"custom_llm_provider": "openai", "model": "gpt-4o"}, None),
        ({"model": "gpt-4o"}, None),
    ],
    ids=["explicit hosted_vllm", "model prefix", "explicit openai", "openai model"],
)
def test_litellm_executed_provider_of(credentials: Mapping[str, object], expected: str | None) -> None:
    assert litellm_executed_provider_of(credentials) == expected


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"custom_llm_provider": "anthropic", "model": "claude-sonnet-4-5"}, "anthropic"),
        ({"model": "anthropic/claude-sonnet-4-5"}, "anthropic"),
        ({"custom_llm_provider": "openai", "model": "gpt-4o"}, None),
        ({"custom_llm_provider": "hosted_vllm", "model": "hosted_vllm/qwen"}, None),
        ({"model": "gpt-4o"}, None),
    ],
    ids=["explicit anthropic", "model prefix", "explicit openai", "executed-only provider", "openai model"],
)
def test_litellm_stored_batch_input_provider_of(credentials: Mapping[str, object], expected: str | None) -> None:
    assert litellm_stored_batch_input_provider_of(credentials) == expected


def test_litellm_stored_batch_input_provider_of_never_probes_the_upstream() -> None:
    assert litellm_stored_batch_input_provider_of({"model": "anthropic/claude-sonnet-4-5"}) == "anthropic"


VLLM_CREDENTIALS: Final[Mapping[str, object]] = {
    "model": "hosted_vllm/qwen",
    "api_base": "http://vllm.test/v1/",
    "api_key": "vllm-key",
}


@dataclass(slots=True)
class FakeFilesApiProbe:
    lacks_files_api: bool
    upstreams: list[tuple[str, str | None]]

    async def __call__(self, api_base: str, api_key: str | None) -> bool:
        self.upstreams.append((api_base, api_key))
        return self.lacks_files_api


@dataclass(slots=True)
class FakeHttpGetter:
    outcome: int | httpx.HTTPError
    requests: list[tuple[str, dict[str, str] | None]]

    async def get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float | httpx.Timeout | None = None
    ) -> httpx.Response:
        self.requests.append((url, headers))
        if isinstance(self.outcome, httpx.HTTPError):
            raise self.outcome
        return httpx.Response(self.outcome)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (404, True),
        (200, False),
        (405, False),
        (401, False),
        (500, False),
        (httpx.ConnectError("refused"), False),
        (httpx.ReadTimeout("slow"), False),
    ],
    ids=["no files route", "lists files", "files route without list", "unauthorized", "server error", "down", "slow"],
)
async def test_upstream_lacks_files_api_only_when_the_files_route_is_a_404(
    outcome: int | httpx.HTTPError, expected: bool
) -> None:
    assert await upstream_lacks_files_api("http://vllm.test/v1", "vllm-key", FakeHttpGetter(outcome, [])) is expected


@pytest.mark.parametrize(
    ("api_base", "api_key", "expected_headers"),
    [
        ("http://vllm.test/v1/", "vllm-key", {"Authorization": "Bearer vllm-key"}),
        ("http://vllm.test/v1", None, None),
    ],
    ids=["trailing slash with key", "keyless"],
)
async def test_upstream_lacks_files_api_asks_the_files_route_under_the_api_base(
    api_base: str, api_key: str | None, expected_headers: dict[str, str] | None
) -> None:
    http_client = FakeHttpGetter(404, [])
    await upstream_lacks_files_api(api_base, api_key, http_client)
    assert http_client.requests == [("http://vllm.test/v1/files", expected_headers)]


@pytest.mark.parametrize(
    ("lacks_files_api", "expected"), [(True, "hosted_vllm"), (False, None)], ids=["bare", "router"]
)
async def test_litellm_executed_provider_for_leaves_a_server_with_its_own_files_api_alone(
    lacks_files_api: bool, expected: str | None
) -> None:
    probe = FakeFilesApiProbe(lacks_files_api, [])
    assert await litellm_executed_provider_for(VLLM_CREDENTIALS, probe) == expected
    assert probe.upstreams == [("http://vllm.test/v1/", "vllm-key")]


@pytest.mark.parametrize(
    "credentials",
    [{"custom_llm_provider": "openai", "model": "gpt-4o", "api_base": "http://openai.test/v1"}, {"model": 7}],
    ids=["provider runs its own batches", "no model to resolve an api_base from"],
)
async def test_litellm_executed_provider_for_never_probes_what_it_would_not_run(
    credentials: Mapping[str, object],
) -> None:
    probe = FakeFilesApiProbe(True, [])
    assert await litellm_executed_provider_for(credentials, probe) is None
    assert probe.upstreams == []


@pytest.mark.parametrize(
    ("credentials", "expected"), [(None, None), (VLLM_CREDENTIALS, "hosted_vllm")], ids=["unknown", "vllm"]
)
async def test_resolve_litellm_executed_provider_asks_the_router_for_the_team_scoped_deployment(
    credentials: Mapping[str, object] | None, expected: str | None
) -> None:
    router = MagicMock(spec=Router)
    router.get_deployment_credentials_with_provider.return_value = credentials
    assert (
        await resolve_litellm_executed_provider(router, BATCH_MODEL, "team-1", FakeFilesApiProbe(True, [])) == expected
    )
    router.get_deployment_credentials_with_provider.assert_called_once_with(model_id=BATCH_MODEL, team_id="team-1")


async def test_create_stores_a_validating_batch_and_completes_it_in_the_background() -> None:
    harness = make_runner()
    created, finished = await harness.create_and_finish()

    assert created.status == "validating"
    assert is_litellm_executed_batch(created.id)
    assert created.id.startswith(f"litellm_proxy;model_id:{DEPLOYMENT_ID};llm_batch_id:litellm_batch_")
    assert (created.model, created.input_file_id) == (BATCH_MODEL, INPUT_FILE_ID)
    assert created.request_counts == BatchRequestCounts(completed=0, failed=0, total=2)
    first_write = harness.store.calls[0]
    assert (first_write.unified_object_id, first_write.model_object_id) == (
        created.id,
        get_batch_id_from_unified_batch_id(created.id),
    )
    assert (first_write.persist_attribution, first_write.batch_processed, first_write.request_tags) == (
        True,
        True,
        ("tag-a",),
    )
    assert harness.storage_factory.calls == [(STORAGE_BACKEND, harness.prisma)]
    assert harness.storage.downloads == [STORAGE_URL]

    assert finished.status == "completed"
    assert finished.request_counts == BatchRequestCounts(completed=2, failed=0, total=2)
    assert (finished.output_file_id, finished.error_file_id) == ("unified-output-1", None)
    assert finished.in_progress_at is not None
    assert finished.completed_at is not None


async def test_create_dispatches_each_row_with_the_batch_model_and_the_key_metadata() -> None:
    harness = make_runner()
    created, _ = await harness.create_and_finish()

    calls = {call.kwargs["messages"][0]["content"]: call.kwargs for call in harness.router.acompletion.await_args_list}
    assert set(calls) == {"hi 1", "hi 2"}
    for content, kwargs in calls.items():
        assert kwargs["model"] == BATCH_MODEL
        assert kwargs["messages"] == [{"role": "user", "content": content}]
        metadata = kwargs["metadata"]
        assert metadata["user_api_key"] == harness.user.api_key
        assert metadata["tags"] == ["tag-a"]
        assert metadata["batch_id"] == created.id
        assert metadata["user_api_key_user_id"] == "user-1"
        assert metadata["user_api_key_team_id"] == "team-1"
        assert metadata["user_api_key_alias"] == "alias-1"
        assert metadata["user_api_key_user_email"] == "user@example.com"


async def test_create_uploads_one_output_line_per_row_with_the_router_response() -> None:
    harness = make_runner()
    replies = {"hi 1": chat_response("hi 1"), "hi 2": chat_response("hi 2")}
    harness.router.acompletion.side_effect = lambda **kwargs: replies[kwargs["messages"][0]["content"]]
    created, _ = await harness.create_and_finish()

    assert len(harness.uploads.calls) == 1
    upload = harness.uploads.calls[0]
    assert (upload.target_storage, upload.purpose, upload.target_model_names) == (
        "litellm_db",
        "batch_output",
        (BATCH_MODEL,),
    )
    assert upload.filename == f"{get_batch_id_from_unified_batch_id(created.id)}_output.jsonl"
    assert upload.user_api_key_dict is harness.user
    assert upload.prisma_client is harness.prisma
    lines = upload.lines()
    assert set(lines) == {"row-1", "row-2"}
    for custom_id, content in (("row-1", "hi 1"), ("row-2", "hi 2")):
        line = lines[custom_id]
        assert str(line["id"]).startswith("batch_req_")
        assert line["error"] is None
        response = line["response"]
        assert isinstance(response, dict)
        assert response["status_code"] == 200
        assert response["body"] == replies[content].model_dump(mode="json")


async def test_create_splits_failed_rows_into_the_error_file() -> None:
    harness = make_runner()
    failure = ProviderRateLimited("slow down")
    reply = chat_response("hi 1")

    def dispatch(messages: Sequence[Mapping[str, str]], **_: object) -> ModelResponse:
        if messages[0]["content"] == "hi 1":
            return reply
        raise failure

    harness.router.acompletion.side_effect = dispatch
    created, finished = await harness.create_and_finish()

    assert finished.status == "completed"
    assert finished.request_counts == BatchRequestCounts(completed=1, failed=1, total=2)
    assert (finished.output_file_id, finished.error_file_id) == ("unified-output-1", "unified-output-2")
    llm_batch_id = get_batch_id_from_unified_batch_id(created.id)
    assert [call.filename for call in harness.uploads.calls] == [
        f"{llm_batch_id}_output.jsonl",
        f"{llm_batch_id}_error.jsonl",
    ]
    assert set(harness.uploads.calls[0].lines()) == {"row-1"}
    error_lines = harness.uploads.calls[1].lines()
    assert set(error_lines) == {"row-2"}
    response = error_lines["row-2"]["response"]
    assert isinstance(response, dict)
    assert response["status_code"] == 429
    assert response["body"] == {
        "error": {"message": str(failure), "type": "ProviderRateLimited", "param": None, "code": None}
    }


async def test_create_rejects_an_unsupported_endpoint() -> None:
    harness = make_runner()
    with pytest.raises(ProxyException) as raised:
        await harness.create(endpoint="/v1/moderations")
    assert raised.value.code == "400"
    assert raised.value.type == "invalid_request_error"
    assert "/v1/moderations" in raised.value.message
    assert harness.store.calls == []
    assert harness.storage_factory.calls == []


@pytest.mark.parametrize(
    "files",
    [{}, {INPUT_FILE_ID: managed_input_file(storage_backend=None)}],
    ids=["unknown file", "no stored content"],
)
async def test_create_rejects_an_input_file_litellm_does_not_hold(
    files: Mapping[str, LiteLLM_ManagedFileTable],
) -> None:
    harness = make_runner(files=files)
    with pytest.raises(ProxyException) as raised:
        await harness.create()
    assert raised.value.code == "400"
    assert "POST /v1/files" in raised.value.message
    assert harness.storage_factory.calls == []
    assert harness.store.calls == []


async def test_create_rejects_an_invalid_input_file() -> None:
    harness = make_runner(content=jsonl(chat_row("a", "hi"), chat_row("a", "again")))
    with pytest.raises(ProxyException) as raised:
        await harness.create()
    assert raised.value.code == "400"
    assert raised.value.message.startswith("Invalid batch input file:")
    assert "'a'" in raised.value.message
    assert harness.store.calls == []


async def test_create_surfaces_a_storage_backend_error_as_a_400() -> None:
    harness = make_runner(storage_error=ValueError("Unknown storage backend 's3'"))
    with pytest.raises(ProxyException) as raised:
        await harness.create()
    assert raised.value.code == "400"
    assert raised.value.message == "Unknown storage backend 's3'"
    assert harness.store.calls == []


CREDENTIAL_ROWS: Final = jsonl(chat_row("row-1", "hi 1"), chat_row("row-2", "hi 2", api_base="https://evil.example"))


async def test_create_rejects_a_row_carrying_client_side_credentials() -> None:
    harness = make_runner(content=CREDENTIAL_ROWS)
    with pytest.raises(ProxyException) as raised:
        await harness.create()
    assert raised.value.code == "400"
    assert raised.value.message.startswith("Invalid batch input file: line 2")
    assert "api_base" in raised.value.message
    assert "allow_client_side_credentials" in raised.value.message
    assert harness.store.calls == []
    assert harness.router.acompletion.await_count == 0


async def test_create_forwards_row_credentials_when_the_admin_opted_in() -> None:
    harness = make_runner(
        content=CREDENTIAL_ROWS, general_settings=MappingProxyType({"allow_client_side_credentials": True})
    )
    _, finished = await harness.create_and_finish()

    assert finished.status == "completed"
    assert finished.request_counts == BatchRequestCounts(completed=2, failed=0, total=2)
    by_content = {
        call.kwargs["messages"][0]["content"]: call.kwargs for call in harness.router.acompletion.await_args_list
    }
    assert by_content["hi 2"]["api_base"] == "https://evil.example"
    assert "api_base" not in by_content["hi 1"]


async def test_running_batch_touches_its_row_until_it_finishes() -> None:
    harness = make_runner(heartbeat_seconds=0.01)

    async def slow_dispatch(**_: object) -> ModelResponse:
        await asyncio.sleep(0.05)
        return chat_response("slow")

    harness.router.acompletion.side_effect = slow_dispatch
    created, finished = await harness.create_and_finish()

    touches = harness.table.touches
    assert finished.status == "completed"
    assert touches
    assert set(touches) == {(created.id, "user-1")}
    assert [call.status for call in harness.store.calls] == ["validating"]
    assert harness.written_statuses() == ["in_progress", "finalizing", "completed"]
    beats_at_finish = len(touches)
    await asyncio.sleep(0.05)
    assert len(touches) == beats_at_finish


async def test_fail_abandoned_marks_a_stale_batch_failed_with_the_runner_lost_error() -> None:
    harness = make_runner()
    batch = seeded_batch(harness.store, "in_progress", age=STALE)

    failed = await harness.runner.fail_abandoned(batch, harness.user)

    assert failed.status == "failed"
    assert failed.failed_at is not None
    assert failed.errors is not None
    assert [(error.message, error.code) for error in failed.errors.data or []] == [
        (litellm_executed_batches._RUNNER_LOST_MESSAGE, "runner_lost")
    ]
    assert harness.store.batch(batch.id).status == "failed"
    assert harness.store.calls == []
    assert harness.table.writes == [StatusWrite(batch.id, "failed", STATUS_WRITE_COLUMNS)]


async def test_fail_abandoned_leaves_a_batch_that_finished_after_the_stale_read() -> None:
    harness = make_runner()
    stale_read = seeded_batch(harness.store, "in_progress", age=STALE)
    harness.store.write(stale_read.model_copy(update={"status": "completed", "output_file_id": "out-1"}), age=STALE)

    current = await harness.runner.fail_abandoned(stale_read, harness.user)

    assert (current.status, current.output_file_id) == ("completed", "out-1")
    assert harness.store.batch(stale_read.id).status == "completed"
    assert harness.table.writes == []


async def test_fail_abandoned_leaves_a_batch_its_runner_touched_since_the_read() -> None:
    harness = make_runner()
    batch = seeded_batch(harness.store, "in_progress", age=STALE)
    harness.store.write(batch)

    current = await harness.runner.fail_abandoned(batch, harness.user)

    assert current.status == "in_progress"
    assert harness.store.batch(batch.id).status == "in_progress"
    assert harness.table.writes == []


async def test_run_does_not_reverse_a_failure_written_between_its_read_and_its_completed_write() -> None:
    harness = make_runner()

    def fail_once_finalizing_is_read(row: StoredObject | None) -> None:
        if row is not None and row.status == "finalizing":
            harness.store.write(row.batch().model_copy(update={"status": "failed"}))

    harness.table.after_read = fail_once_finalizing_is_read
    _, finished = await harness.create_and_finish()

    assert finished.status == "failed"
    assert finished.output_file_id is None
    assert harness.written_statuses() == ["in_progress", "finalizing"]


async def test_run_honours_a_cancel_written_between_its_read_and_its_finalizing_write() -> None:
    harness = make_runner(content=jsonl(chat_row("row-1", "hi 1")))

    def cancel_once_the_row_is_dispatched(row: StoredObject | None) -> None:
        if row is not None and row.status == "in_progress" and harness.router.acompletion.await_count == 1:
            harness.store.write(row.batch().model_copy(update={"status": "cancelling"}))

    harness.table.after_read = cancel_once_the_row_is_dispatched
    _, finished = await harness.create_and_finish()

    assert finished.status == "cancelled"
    assert finished.request_counts == BatchRequestCounts(completed=1, failed=0, total=1)
    assert finished.output_file_id == "unified-output-1"
    assert harness.written_statuses() == ["in_progress", "cancelling", "cancelled"]


async def test_running_batch_stops_and_writes_nothing_once_a_retriever_marked_it_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_executed_batches, "_CANCEL_POLL_SECONDS", 0.0)
    rows = jsonl(chat_row("row-1", "hi 1"), chat_row("row-2", "hi 2"), chat_row("row-3", "hi 3"))
    harness = make_runner(content=rows, concurrency=1)

    def dispatch(metadata: Mapping[str, object], **_: object) -> ModelResponse:
        running = harness.store.batch(str(metadata["batch_id"]))
        harness.store.write(running.model_copy(update={"status": "failed"}))
        return chat_response("hi 1")

    harness.router.acompletion.side_effect = dispatch
    _, finished = await harness.create_and_finish()

    assert harness.router.acompletion.await_count == 1
    assert finished.status == "failed"
    assert [call.status for call in harness.store.calls] == ["validating"]
    assert harness.written_statuses() == ["in_progress"]
    assert harness.uploads.calls == []


@pytest.mark.parametrize(
    ("endpoint", "body", "method"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, "acompletion"),
        ("/v1/completions", {"prompt": "hi"}, "atext_completion"),
        ("/v1/embeddings", {"input": "hi"}, "aembedding"),
        ("/v1/responses", {"input": "hi"}, "aresponses"),
    ],
)
async def test_each_endpoint_awaits_only_its_router_method(
    endpoint: BatchEndpoint, body: Mapping[str, object], method: str
) -> None:
    row = {"custom_id": "a", "method": "POST", "url": endpoint, "body": {"model": "row-model", **body}}
    harness = make_runner(content=jsonl(row))
    _, finished = await harness.create_and_finish(endpoint)

    assert finished.request_counts == BatchRequestCounts(completed=1, failed=0, total=1)
    awaited = {name: getattr(harness.router, name).await_count for name in ROUTER_METHODS}
    assert awaited == {name: int(name == method) for name in ROUTER_METHODS}
    kwargs = getattr(harness.router, method).await_args.kwargs
    assert kwargs["model"] == BATCH_MODEL
    assert kwargs["disable_fallbacks"] is True
    assert all(kwargs[key] == value for key, value in body.items())


async def test_cancel_unknown_batch_is_404() -> None:
    harness = make_runner()
    with pytest.raises(ProxyException) as raised:
        await harness.runner.cancel("missing-batch", harness.user)
    assert raised.value.code == "404"


async def test_cancel_terminal_batch_is_400() -> None:
    harness = make_runner()
    batch = seeded_batch(harness.store, "completed")
    with pytest.raises(ProxyException) as raised:
        await harness.runner.cancel(batch.id, harness.user)
    assert raised.value.code == "400"
    assert "completed" in raised.value.message
    assert harness.table.writes == []


async def test_cancel_marks_a_running_batch_cancelling_once() -> None:
    harness = make_runner()
    batch = seeded_batch(harness.store, "in_progress")

    cancelled = await harness.runner.cancel(batch.id, harness.user)

    assert cancelled.status == "cancelling"
    assert cancelled.cancelling_at is not None
    assert harness.store.batch(batch.id).status == "cancelling"
    assert harness.store.calls == []
    assert harness.table.writes == [StatusWrite(batch.id, "cancelling", STATUS_WRITE_COLUMNS)]

    again = await harness.runner.cancel(batch.id, harness.user)

    assert again.model_dump() == cancelled.model_dump()
    assert len(harness.table.writes) == 1


async def test_cancel_racing_a_completion_is_400_and_leaves_the_batch_completed() -> None:
    harness = make_runner()
    batch = seeded_batch(harness.store, "in_progress")

    def complete_once_read(row: StoredObject | None) -> None:
        if row is not None and row.status == "in_progress":
            harness.store.write(row.batch().model_copy(update={"status": "completed"}))

    harness.table.after_read = complete_once_read
    with pytest.raises(ProxyException) as raised:
        await harness.runner.cancel(batch.id, harness.user)

    assert raised.value.code == "400"
    assert harness.store.batch(batch.id).status == "completed"
    assert harness.table.writes == []


async def test_running_batch_skips_the_remaining_rows_after_an_operator_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_executed_batches, "_CANCEL_POLL_SECONDS", 0.0)
    rows = jsonl(chat_row("row-1", "hi 1"), chat_row("row-2", "hi 2"), chat_row("row-3", "hi 3"))
    harness = make_runner(content=rows, concurrency=1)
    reply = chat_response("hi 1")

    def dispatch(metadata: Mapping[str, object], **_: object) -> ModelResponse:
        running = harness.store.batch(str(metadata["batch_id"]))
        harness.store.write(running.model_copy(update={"status": "cancelling"}))
        return reply

    harness.router.acompletion.side_effect = dispatch
    _, finished = await harness.create_and_finish()

    assert harness.router.acompletion.await_count == 1
    assert finished.status == "cancelled"
    assert finished.cancelled_at is not None
    assert finished.request_counts == BatchRequestCounts(completed=1, failed=0, total=3)
    assert (finished.output_file_id, finished.error_file_id) == ("unified-output-1", None)


async def test_batch_expires_at_the_completion_window_and_keeps_what_finished() -> None:
    rows = jsonl(chat_row("row-1", "hi 1"), chat_row("row-2", "hi 2"), chat_row("row-3", "hi 3"))
    harness = make_runner(content=rows, concurrency=1, completion_window_seconds=0.2)
    reply = chat_response("hi 1")

    async def dispatch(messages: Sequence[Mapping[str, str]], **_: object) -> ModelResponse:
        if messages[0]["content"] == "hi 1":
            return reply
        await asyncio.Event().wait()
        raise AssertionError("a row still running at the completion window must be cut off")

    harness.router.acompletion.side_effect = dispatch
    created, finished = await harness.create_and_finish()

    assert created.expires_at == created.created_at
    assert finished.status == "expired"
    assert finished.expired_at is not None
    assert finished.request_counts == BatchRequestCounts(completed=1, failed=2, total=3)
    assert (finished.output_file_id, finished.error_file_id) == ("unified-output-1", "unified-output-2")
    assert set(harness.uploads.calls[0].lines()) == {"row-1"}
    error_lines = harness.uploads.calls[1].lines()
    assert set(error_lines) == {"row-2", "row-3"}
    for line in error_lines.values():
        assert line["response"] is None
        error = line["error"]
        assert isinstance(error, dict)
        assert error["code"] == "batch_expired"


async def test_a_provider_timeout_fails_its_row_without_expiring_the_batch() -> None:
    harness = make_runner()
    reply = chat_response("hi 2")

    def dispatch(messages: Sequence[Mapping[str, str]], **_: object) -> ModelResponse:
        if messages[0]["content"] == "hi 1":
            raise asyncio.TimeoutError("the provider took too long")
        return reply

    harness.router.acompletion.side_effect = dispatch
    _, finished = await harness.create_and_finish()

    assert finished.status == "completed"
    assert finished.expired_at is None
    assert finished.request_counts == BatchRequestCounts(completed=1, failed=1, total=2)
    assert set(harness.uploads.calls[0].lines()) == {"row-2"}
    error_lines = harness.uploads.calls[1].lines()
    assert set(error_lines) == {"row-1"}
    assert error_lines["row-1"]["error"] is None
    response = error_lines["row-1"]["response"]
    assert isinstance(response, dict)
    assert response["status_code"] == 500
    assert response["body"] == {
        "error": {"message": "the provider took too long", "type": "TimeoutError", "param": None, "code": None}
    }


async def test_batch_created_past_its_window_dispatches_nothing() -> None:
    harness = make_runner(completion_window_seconds=0)
    _, finished = await harness.create_and_finish()

    assert harness.router.acompletion.await_count == 0
    assert finished.status == "expired"
    assert finished.request_counts == BatchRequestCounts(completed=0, failed=2, total=2)
    assert (finished.output_file_id, finished.error_file_id) == (None, "unified-output-1")
    assert set(harness.uploads.calls[0].lines()) == {"row-1", "row-2"}


async def test_upload_failure_marks_the_batch_failed() -> None:
    harness = make_runner(upload_error=RuntimeError("storage exploded"))
    _, finished = await harness.create_and_finish()

    assert finished.status == "failed"
    assert finished.failed_at is not None
    assert finished.output_file_id is None
    assert finished.errors is not None
    assert [(error.message, error.code) for error in finished.errors.data or []] == [
        ("storage exploded", "internal_error")
    ]


async def test_only_the_create_write_carries_attribution_and_billing_flags() -> None:
    harness = make_runner()
    await harness.create_and_finish()

    assert [(call.status, call.persist_attribution, call.batch_processed) for call in harness.store.calls] == [
        ("validating", True, True)
    ]
    assert [write.columns for write in harness.table.writes] == [STATUS_WRITE_COLUMNS] * 3


async def test_run_completes_under_the_real_hooks_base64_batch_id() -> None:
    harness = make_runner(store_factory=RealIdManagedBatchStore)
    created, finished = await harness.create_and_finish()

    assert _is_base64_encoded_unified_file_id(created.id)
    assert finished.status == "completed"
    assert [call.model_object_id.startswith("litellm_batch_") for call in harness.store.calls] == [True]
    assert [write.unified_object_id for write in harness.table.writes] == [created.id] * 3


async def test_cancel_works_under_the_real_hooks_base64_batch_id() -> None:
    harness = make_runner(store_factory=RealIdManagedBatchStore)
    batch = seeded_batch(harness.store, "in_progress")
    cancelled = await harness.runner.cancel(batch.id, harness.user)

    assert cancelled.status == "cancelling"
    assert harness.store.batch(batch.id).status == "cancelling"
    assert [write.unified_object_id for write in harness.table.writes] == [batch.id]
