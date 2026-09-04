"""Live e2e for the Batches API across every provider LiteLLM supports.

Mostly synchronous tier: a batch's completion window is 24h, so the lifecycle
matrix never waits for "completed". Each case uploads a tiny JSONL, creates the
batch through one of the four routing scenarios, asserts it was accepted
(non-terminal status) and routed to the right provider, then retrieves / cancels /
lists where the provider supports it. Everything created is deleted on teardown.
The exception is TestBatchTerminalState, which carries completed-state + cost
write-back coverage via a cross-run marker baton (design in COVERAGE.md).

Routing signal: for provider_fallback the raw batch id discriminates the provider;
for the encoded/unified/model_param scenarios the proxy re-encodes the id, so the
load-bearing signal is that create SUCCEEDS against that provider's own model - a
misroute to the wrong provider fails the create.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import pytest
from pydantic import BaseModel

from e2e_config import PROXY_BASE_URL, unique_marker

from batch_client import (
    UPLOAD_FILENAME,
    BatchClient,
    BatchCreateBody,
    BatchObject,
    FileObject,
    is_model_access_denied,
    is_result_access_denied,
)
from capabilities import (
    AZURE_BATCH_MODEL,
    BATCH_ID_SHAPE,
    CAPABILITIES,
    FILE_ID_SHAPE,
    OPENAI_BATCH_BACKEND,
    OPENAI_BATCH_MODEL,
    PROVIDERS,
    Capability,
    Provider,
    batch_model_name,
    coverage_cells_for_lifecycle,
    decoded_model_from_id,
    is_managed_id,
    matches_id_shape,
    openai_batch_params,
    raw_id_matches_provider,
)
from e2e_http import (
    FileUploadForm,
    Result,
    StreamingResponse,
    Success,
    UnknownApiError,
    require_successful_call,
    unwrap,
)
from lifecycle import ResourceManager
from models import KeyGenerateBody, KeyMetadata, LiteLLMParamsBody, SpendLogRow

pytestmark = pytest.mark.e2e

CREATED_BATCH_STATUSES = {"validating", "in_progress", "finalizing"}
BATCH_CANCEL_DELAY_SECONDS = 2
BATCH_TERMINAL_BEFORE_CANCEL = {"failed", "cancelled", "expired"}
BATCH_OP_RETRIES = 5
# Azure / Vertex cancel and the pre-cancel re-retrieve are provider-side flakes
# (connection refused, brief 500s) and the registry only has one basic cell per
# provider (shared across scenarios). Create + retrieve already prove routing;
# cancel is still deferred for cleanup, just not asserted for these two.
_CANCEL_ASSERTED_PROVIDERS = frozenset({"openai"})


def _transient_status(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


def _backoff_seconds(attempt: int) -> float:
    delays: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 8.0)
    return delays[min(attempt, len(delays) - 1)]


def cancel_batch(
    client: BatchClient, batch_id: str, *, key: str, provider: str | None
) -> BatchObject:
    last = client.cancel_batch(batch_id, key=key, provider=provider)
    for attempt in range(BATCH_OP_RETRIES - 1):
        match last:
            case Success(data=data):
                return data
            case UnknownApiError(status_code=code) if _transient_status(code):
                time.sleep(_backoff_seconds(attempt))
                last = client.cancel_batch(batch_id, key=key, provider=provider)
            case _:
                break
    return unwrap(last)


def retrieve_batch(
    client: BatchClient, batch_id: str, *, key: str, provider: str | None
) -> BatchObject:
    last = client.retrieve_batch(batch_id, key=key, provider=provider)
    for attempt in range(BATCH_OP_RETRIES - 1):
        match last:
            case Success(data=data):
                return data
            case UnknownApiError(status_code=code) if _transient_status(code):
                time.sleep(_backoff_seconds(attempt))
                last = client.retrieve_batch(batch_id, key=key, provider=provider)
            case _:
                break
    return unwrap(last)


def create_batch_resilient(
    client: BatchClient, cap: Capability, file_id: str, key: str
) -> StreamingResponse:
    last = create_for_scenario(client, cap, file_id, key)
    for attempt in range(BATCH_OP_RETRIES - 1):
        if last.ok:
            return last
        if not _transient_status(last.status_code):
            return last
        time.sleep(_backoff_seconds(attempt))
        last = create_for_scenario(client, cap, file_id, key)
    return last


def render_jsonl(model: str) -> bytes:
    line = {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        },
    }
    return (json.dumps(line) + "\n").encode()


def upload_for_scenario(
    client: BatchClient, cap: Capability, content: bytes, key: str
) -> Result[FileObject]:
    if cap.scenario == "encoded":
        return client.upload_file(
            content=content,
            form=FileUploadForm(purpose="batch"),
            model=cap.model,
            key=key,
        )
    if cap.scenario == "unified":
        return client.upload_file(
            content=content,
            form=FileUploadForm(purpose="batch", target_model_names=cap.model),
            key=key,
        )
    return client.upload_file(
        content=content,
        form=FileUploadForm(purpose="batch"),
        key=key,
        provider=cap.provider,
    )


def create_for_scenario(
    client: BatchClient, cap: Capability, file_id: str, key: str
) -> StreamingResponse:
    if cap.scenario == "model_param":
        return client.create_batch(
            body=BatchCreateBody(input_file_id=file_id, model=cap.model), key=key
        )
    if cap.scenario == "provider_fallback":
        return client.create_batch(
            body=BatchCreateBody(input_file_id=file_id), key=key, provider=cap.provider
        )
    return client.create_batch(body=BatchCreateBody(input_file_id=file_id), key=key)


def op_provider(cap: Capability) -> str | None:
    """provider_fallback ids are raw, so retrieve/cancel/list/delete need the provider
    hint; the other scenarios encode it into the id and route automatically."""
    return cap.provider if cap.scenario == "provider_fallback" else None


def quietly(action: Callable[[], object]) -> Callable[[], None]:
    """Adapt a value-returning call into a best-effort cleanup the teardown can run."""

    def run() -> None:
        action()

    return run


def assert_file_object(file: FileObject, *, provider: str) -> None:
    assert file.object == "file", f"file.object={file.object!r}"
    assert file.purpose == "batch", f"file.purpose={file.purpose!r}"
    assert file.bytes is not None, f"file.bytes={file.bytes!r}"
    if provider != "bedrock":
        assert file.bytes > 0, f"file.bytes={file.bytes!r}"
    assert file.status, "file.status missing"
    assert (
        file.created_at is not None and file.created_at > 0
    ), "file.created_at missing"


def assert_batch_object(batch: BatchObject) -> None:
    assert batch.object == "batch", f"batch.object={batch.object!r}"
    if batch.endpoint:
        assert (
            batch.endpoint == "/v1/chat/completions"
        ), f"batch.endpoint={batch.endpoint!r}"
    assert batch.completion_window == "24h", f"window={batch.completion_window!r}"
    assert batch.input_file_id, "batch.input_file_id missing"
    assert (
        batch.created_at is not None and batch.created_at > 0
    ), "batch.created_at missing"


@pytest.mark.parametrize(
    "cap",
    [
        pytest.param(
            cap,
            id=cap.id,
            marks=pytest.mark.covers(*coverage_cells_for_lifecycle(cap)),
        )
        for cap in CAPABILITIES
    ],
)
def test_batch_lifecycle(
    cap: Capability,
    client: BatchClient,
    resources: ResourceManager,
    batch_deployments: None,
) -> None:
    key = resources.key()
    provider = op_provider(cap)

    file = unwrap(upload_for_scenario(client, cap, render_jsonl(cap.jsonl_model), key))
    resources.defer(
        quietly(lambda: client.delete_file(file.id, key=key, provider=provider))
    )
    assert_file_object(file, provider=cap.provider)
    assert matches_id_shape(
        FILE_ID_SHAPE[cap.scenario], file.id
    ), f"{cap.id}: file id {file.id!r} is not a {FILE_ID_SHAPE[cap.scenario]} id"

    created = create_batch_resilient(client, cap, file.id, key)
    require_successful_call(created)
    batch = BatchObject.model_validate_json(created.body)
    resources.defer(
        quietly(lambda: client.cancel_batch(batch.id, key=key, provider=provider))
    )

    assert batch.id, f"create returned no batch id (body={created.body[:200]})"
    assert (
        batch.status in CREATED_BATCH_STATUSES
    ), f"freshly created batch has non-transitional status {batch.status!r}"
    assert_batch_object(batch)
    assert matches_id_shape(
        BATCH_ID_SHAPE[cap.scenario], batch.id
    ), f"{cap.id}: batch id {batch.id!r} is not a {BATCH_ID_SHAPE[cap.scenario]} id"
    if cap.scenario == "provider_fallback":
        assert raw_id_matches_provider(
            cap.provider, batch.id
        ), f"{cap.provider} batch id {batch.id!r} not in that provider's native shape; misrouted?"

    fetched = retrieve_batch(client, batch.id, key=key, provider=provider)
    assert_batch_object(fetched)
    assert fetched.id == batch.id
    assert (
        fetched.input_file_id == batch.input_file_id
    ), "retrieve changed input_file_id"
    assert fetched.status, "retrieved batch has no status"

    if cap.can_cancel and cap.provider in _CANCEL_ASSERTED_PROVIDERS:
        time.sleep(BATCH_CANCEL_DELAY_SECONDS)
        pre_cancel = retrieve_batch(client, batch.id, key=key, provider=provider)
        assert (
            pre_cancel.status not in BATCH_TERMINAL_BEFORE_CANCEL
        ), (
            f"batch reached {pre_cancel.status!r} before cancel; "
            "provider likely rejected the input"
        )
        if pre_cancel.status == "completed":
            return
        cancelled = cancel_batch(client, batch.id, key=key, provider=provider)
        assert cancelled.id == batch.id
        assert cancelled.object == "batch"
        assert cancelled.status in {"cancelling", "cancelled"}, (
            f"unexpected post-cancel status {cancelled.status!r}"
        )

    if cap.can_list:
        list_result = client.list_batches(key=key, provider=provider)
        managed_filter_unsupported = False
        match list_result:
            case UnknownApiError(body=body) if (
                "Filtering by 'provider' is not supported when using managed batches" in body
            ):
                managed_filter_unsupported = True
                listed = unwrap(client.list_batches(key=key, provider=None))
            case _:
                listed = unwrap(list_result)
        if listed.object is not None:
            assert listed.object == "list", f"list envelope object={listed.object!r}"
        match = next((b for b in listed.data if b.id == batch.id), None)
        if (
            match is None
            and managed_filter_unsupported
            and cap.scenario == "provider_fallback"
        ):
            # provider_fallback keeps the provider's raw batch id (not re-encoded
            # into a managed/proxy id). When the gateway rejects provider-scoped
            # list, the only available list is the unfiltered managed view, which
            # does not index raw provider ids. Membership cannot be asserted here;
            # create + retrieve (and raw_id_matches_provider above) already pin
            # routing for this scenario.
            return
        assert match is not None, "created batch absent from list"
        assert match.object == "batch"


@pytest.mark.covers("llm.batches.openai.key_model_access_denied.nonstream.works")
def test_batch_key_model_access_denied(
    client: BatchClient, resources: ResourceManager, batch_deployments: None
) -> None:
    key = resources.key(models=[OPENAI_BATCH_MODEL])

    denied_upload = client.upload_file(
        content=render_jsonl(AZURE_BATCH_MODEL),
        form=FileUploadForm(purpose="batch"),
        model=AZURE_BATCH_MODEL,
        key=key,
    )
    assert is_result_access_denied(
        denied_upload
    ), f"restricted key uploaded a file for a disallowed model: {denied_upload}"

    raw_file = unwrap(
        client.upload_file(
            content=render_jsonl(OPENAI_BATCH_MODEL),
            form=FileUploadForm(purpose="batch"),
            key=key,
            provider="openai",
        )
    ).id
    resources.defer(
        quietly(lambda: client.delete_file(raw_file, key=key, provider="openai"))
    )

    denied_create = client.create_batch(
        body=BatchCreateBody(input_file_id=raw_file, model=AZURE_BATCH_MODEL), key=key
    )
    assert is_model_access_denied(
        denied_create
    ), f"restricted key created a batch for a disallowed model (status {denied_create.status_code})"


@pytest.mark.covers(
    "llm.files.openai.upload.nonstream.works",
    "llm.files.openai.delete.nonstream.works",
)
def test_file_upload_and_delete_outputs(
    client: BatchClient, resources: ResourceManager, batch_deployments: None
) -> None:
    key = resources.key()
    file = unwrap(
        client.upload_file(
            content=render_jsonl(OPENAI_BATCH_MODEL),
            form=FileUploadForm(purpose="batch"),
            model=OPENAI_BATCH_MODEL,
            key=key,
        )
    )
    assert_file_object(file, provider="openai")

    deleted = unwrap(client.delete_file(file.id, key=key))
    assert deleted.id, "delete response has no id"
    assert deleted.object == "file", f"delete object={deleted.object!r}"
    assert deleted.deleted is True, "file was not reported deleted"


def unattributed_rows(rows: list[SpendLogRow]) -> list[SpendLogRow]:
    """Spend rows that carry no caller identity (empty api_key).

    Every request the proxy bills is stamped with the calling key. A row with no
    api_key is one the proxy could not attribute; LIT-3266 is exactly this: the
    batch rate limiter's internal input-file read ran without the batch's auth
    metadata, landing a spend row with empty api_key/user. The symptom is not
    tied to a single call_type, so this catches any unattributed row rather than
    only a named file-content one.
    """
    return [row for row in rows if not row.api_key]


@pytest.mark.skip(
    reason=(
        "LIT-5027: the path under test hangs. The batch rate limiter reads the input file "
        "to count tokens by awaiting litellm.afile_content with no timeout, so a slow Files "
        "API holds POST /v1/batches open past any client deadline (63.6s observed on stage "
        "against a 60s read timeout). The unattributed-spend-row contract below is never "
        "reached, so the test reports a timeout rather than the behavior it guards. Unskip "
        "once the fetch is bounded."
    )
)
def test_rate_limited_batch_create_leaves_no_unattributed_spend_row(
    client: BatchClient, resources: ResourceManager, batch_deployments: None
) -> None:
    """LIT-3266: creating a batch on a rate-limited key runs the batch rate
    limiter, which reads the input file to count tokens (the limiter only reads
    the file when the key has applicable rpm/tpm limits, so an unlimited key
    hides the path). That internal read must carry the batch's auth metadata;
    the reported gap was that it did not, spawning a spend-log row with empty
    api_key/user. Create returning 200 is not a reliable signal (the read error
    is swallowed), so this asserts the hygiene contract instead: the operation
    introduces no new unattributed spend row.

    The key sets generous rpm/tpm limits (not a restrictive model allowlist) so
    the file-read path fires while the batch itself is not blocked.
    ``resources.key()`` cannot set limits, so the key is minted on the gateway
    directly and its delete deferred.

    Snapshots read /spend/logs/v2 over a bounded window around the test instead
    of the unpaginated /spend/logs whole-table read, which grows with the
    environment and OOMed the e2e runner on stage.
    """
    user_id = f"e2e-batch-rl-{unique_marker()}"
    key = client.proxy.generate_key(
        KeyGenerateBody(models=[], tpm_limit=1_000_000, rpm_limit=1_000, user_id=user_id)
    )
    resources.defer(lambda: client.proxy.delete_key(key))

    window_start = datetime.now(timezone.utc) - timedelta(hours=1)
    window_end = window_start + timedelta(hours=2)
    before = frozenset(
        row.request_id
        for row in unattributed_rows(
            client.proxy.spend_logs_window(start=window_start, end=window_end)
        )
    )

    file = unwrap(
        client.upload_file(
            content=render_jsonl("gpt-4o-mini"),
            form=FileUploadForm(purpose="batch"),
            model=OPENAI_BATCH_MODEL,
            key=key,
        )
    )
    resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

    created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
    require_successful_call(created)
    batch = BatchObject.model_validate_json(created.body)
    resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

    _ = client.proxy.poll_logs_for_key(key, min_rows=1)

    new_orphans = [
        row
        for row in unattributed_rows(
            client.proxy.spend_logs_window(start=window_start, end=window_end)
        )
        if row.request_id not in before
    ]
    assert not new_orphans, (
        "batch create on a rate-limited key left an unattributed spend row "
        f"(LIT-3266); rows={[(r.request_id, r.call_type, r.model) for r in new_orphans]}"
    )


FILE_CONTENT_CELLS = {
    "azure": "llm.files.azure_openai.content.nonstream.works",
    "vertex_ai": "llm.files.vertex.content.nonstream.works",
    "bedrock": "llm.files.bedrock.content.nonstream.works",
}
BYTE_FIDELITY_CONTENT_PROVIDERS = frozenset({"azure"})


class TestBatchFileContent:
    """GET /v1/files/{id}/content returns the uploaded batch JSONL bytes.

    Azure stores the upload verbatim, so its download is asserted byte-equal.
    Vertex (GCS) and Bedrock (S3) transform each JSONL line into the provider's
    request format at upload time, so their downloads assert 200 plus non-empty
    parseable JSON lines instead of byte equality.
    """

    @pytest.mark.covers(
        "llm.files.openai.content.nonstream.works",
        exercised_on=["files"],
    )
    def test_file_content_matches_upload(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        proxy_name = f"e2e-file-content-{unique_marker()}"
        model_id = client.create_model(proxy_name, openai_batch_params())
        resources.defer(lambda: client.delete_model(model_id))
        key = resources.key()

        payload = render_jsonl(OPENAI_BATCH_BACKEND)
        file = unwrap(
            client.upload_file(
                content=payload,
                form=FileUploadForm(purpose="batch", target_model_names=proxy_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert file.id

        downloaded = client.proxy.transport.download(
            f"/v1/files/{file.id}/content",
            headers=client.proxy.transport.bearer(key),
        )
        assert downloaded.status_code == 200, (
            f"file content must be 200, got {downloaded.status_code}: {downloaded.body[:300]}"
        )
        expected = payload.decode().rstrip("\n")
        got = downloaded.body.rstrip("\n")
        assert got == expected, (
            "downloaded file content must match the uploaded JSONL bytes"
        )

    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(
                p,
                id=p.name,
                marks=pytest.mark.covers(
                    FILE_CONTENT_CELLS[p.name], exercised_on=["files"]
                ),
            )
            for p in PROVIDERS
            if p.name in FILE_CONTENT_CELLS
        ],
    )
    def test_unified_file_content_downloads(
        self,
        provider: Provider,
        client: BatchClient,
        resources: ResourceManager,
        batch_deployments: None,
    ) -> None:
        key = resources.key()
        payload = render_jsonl(provider.raw_model)
        file = unwrap(
            client.upload_file(
                content=payload,
                form=FileUploadForm(purpose="batch", target_model_names=provider.model),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert_file_object(file, provider=provider.name)
        assert is_managed_id(file.id), (
            f"{provider.name}: unified upload must return a managed file id, got {file.id!r}"
        )

        downloaded = client.proxy.transport.download(
            f"/v1/files/{file.id}/content",
            headers=client.proxy.transport.bearer(key),
        )
        assert downloaded.status_code == 200, (
            f"{provider.name}: file content must be 200, "
            f"got {downloaded.status_code}: {downloaded.body[:300]}"
        )
        body = downloaded.body.strip()
        assert body, f"{provider.name}: file content download returned an empty body"
        if provider.name in BYTE_FIDELITY_CONTENT_PROVIDERS:
            assert body == payload.decode().strip(), (
                f"{provider.name}: downloaded content must match the uploaded JSONL bytes"
            )
        else:
            for line in body.splitlines():
                assert json.loads(line), (
                    f"{provider.name}: content line is not JSON: {line[:200]}"
                )


class TestOpenAIFiles:
    """GET /v1/files (list) and GET /v1/files/{id} (retrieve) over the OpenAI route.

    The proxy lists the OpenAI org's raw file ids, so the list case uploads a raw
    (provider-routed) file whose id matches what list returns; retrieve re-encodes
    the id it was called with, so the model-encoded upload round-trips unchanged.
    """

    @pytest.mark.covers(
        "llm.files.openai.list.nonstream.works",
        exercised_on=["files"],
    )
    @pytest.mark.skip(
        reason=(
            "LIT-4820 (https://linear.app/litellm-ai/issue/LIT-4820): GET /v1/files omits "
            "newly uploaded files. The upload succeeds and "
            "GET /v1/files/{id} returns the file, but it never appears in the listing; the "
            "returned set is stable with its newest entry ~10h old, on both the managed "
            "(/v1/files?model=) and provider-scoped (/openai/v1/files) routes. Skipped rather "
            "than weakened because the assertion below is the correct contract. Remove this "
            "marker when LIT-4820 is fixed; do not relax the assertion to make it pass."
        )
    )
    def test_uploaded_file_appears_in_list(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        file = unwrap(
            client.upload_file(
                content=render_jsonl(OPENAI_BATCH_MODEL),
                form=FileUploadForm(purpose="batch"),
                key=key,
                provider="openai",
            )
        )
        resources.defer(
            quietly(lambda: client.delete_file(file.id, key=key, provider="openai"))
        )

        listed = unwrap(client.list_files(key=key))
        assert listed.object is None or listed.object == "list", (
            f"list envelope object={listed.object!r}"
        )
        match = next((entry for entry in listed.data if entry.id == file.id), None)
        assert match is not None, f"uploaded file {file.id!r} absent from GET /v1/files"
        assert match.purpose == "batch", (
            f"listed file must round-trip the upload purpose, got {match.purpose!r}"
        )

    @pytest.mark.covers(
        "llm.files.openai.list_isolation.nonstream.works",
        exercised_on=["files"],
    )
    def test_list_page_cursors_address_only_the_callers_own_files(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        """Pins GitHub issue #36087: a list page's pagination cursors must address
        rows in that page.

        The proxy fronts one shared provider account, so the upstream page is the
        whole organization's. The gateway narrows `data` to the files the caller
        owns, and `first_id` / `last_id` have to be narrowed with it: left as the
        upstream org's, they hand any caller raw provider file ids belonging to
        other tenants, which is the handle the file routes accept.
        """
        key = resources.key(user_id=f"e2e-file-list-{unique_marker()}")

        listed = unwrap(client.list_files(key=key))

        expected_first = listed.data[0].id if listed.data else None
        expected_last = listed.data[-1].id if listed.data else None
        assert listed.first_id == expected_first, (
            f"first_id {listed.first_id!r} is not the first row this caller can see "
            f"({expected_first!r}); the page leaked another caller's file id"
        )
        assert listed.last_id == expected_last, (
            f"last_id {listed.last_id!r} is not the last row this caller can see "
            f"({expected_last!r}); the page leaked another caller's file id"
        )
        assert listed.has_more is not True, (
            "the page advertises another page, but the proxy never forwards a cursor "
            "upstream, so following it re-serves this same page forever"
        )

    @pytest.mark.covers(
        "llm.files.openai.retrieve.nonstream.works",
        exercised_on=["files"],
    )
    def test_retrieve_round_trips_metadata(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        file = unwrap(
            client.upload_file(
                content=render_jsonl(OPENAI_BATCH_MODEL),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

        fetched = unwrap(client.retrieve_file(file.id, key=key))
        assert fetched.id == file.id, "retrieve must echo the uploaded file id"
        assert fetched.purpose == "batch", (
            f"retrieve must round-trip purpose, got {fetched.purpose!r}"
        )
        assert fetched.filename == UPLOAD_FILENAME, (
            f"retrieve must round-trip filename, got {fetched.filename!r}"
        )


BATCH_RL_REQUEST_LINES = 3
BATCH_RL_RPM_LIMIT = 2


def _multi_request_jsonl(model: str, n: int) -> bytes:
    lines = tuple(
        json.dumps(
            {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
            }
        )
        for i in range(n)
    )
    return ("\n".join(lines) + "\n").encode()


class TestBatchRateLimitErrorMapping:
    """Batch create that exceeds a key's RPM maps to a structured 429.

    The batch rate limiter reads the input file at submission time and rejects
    the create when the file's request count would exceed the key's remaining
    RPM. The product promise is not only the block itself but the
    OpenAI-compatible shape: HTTP 429, a body that names the batch rate limit,
    and pacing headers so clients can back off. Complements the LIT-3266 hygiene
    check (no orphan spend rows) by asserting the error mapping when the limiter
    actually fires.
    """

    @pytest.mark.covers(
        "quota_management.ratelimit.batch_rpm.blocks_over_limit",
        exercised_on=["batches"],
    )
    def test_batch_create_over_rpm_returns_mapped_429(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        user_id = f"e2e-batch-rl-map-{unique_marker()}"
        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[], rpm_limit=BATCH_RL_RPM_LIMIT, tpm_limit=1_000_000, user_id=user_id
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        file = unwrap(
            client.upload_file(
                content=_multi_request_jsonl("gpt-4o-mini", BATCH_RL_REQUEST_LINES),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)

        assert created.status_code == 429, (
            f"expected batch RPM 429 when file has {BATCH_RL_REQUEST_LINES} requests and "
            f"rpm_limit={BATCH_RL_RPM_LIMIT}, got {created.status_code}: {created.body[:400]}"
        )
        body_lower = created.body.lower()
        assert "batch rate limit exceeded" in body_lower, (
            f"429 body must name the batch rate limit so clients can branch on it; "
            f"got: {created.body[:400]}"
        )
        assert str(BATCH_RL_REQUEST_LINES) in created.body, (
            f"429 body should report the batch request count ({BATCH_RL_REQUEST_LINES}); "
            f"got: {created.body[:400]}"
        )
        assert "rpm" in body_lower or "requests remaining" in body_lower, (
            f"429 body must describe the RPM budget remaining so clients can pace; "
            f"got: {created.body[:400]}"
        )
        retry_after = created.headers.get("retry-after")
        if retry_after is not None:
            assert retry_after.isdigit() and int(retry_after) > 0, (
                f"retry-after must be a positive integer when present, got {retry_after!r}"
            )


BATCH_ENQUEUED_HEADROOM_TOKENS = 100_000
_BATCH_REQUIRES_TOKENS = re.compile(r"Batch requires (\d+) tokens")


class TestBatchEnqueuedTokenLimit:
    """Opt-in enqueued-token allowance governs batch submission instead of RPM/TPM.

    A key whose metadata carries batch_enqueued_token_limit reserves the batch's
    token estimate against that allowance at create time: per-minute limits no
    longer gate batch submission, exhausting the allowance rejects the create
    before it reaches the provider, and cancelling a running batch refunds its
    reservation so blocked submissions go through again (LIT-5273).
    """

    def _upload_batch_file(
        self, client: BatchClient, resources: ResourceManager, key: str
    ) -> FileObject:
        file = unwrap(
            client.upload_file(
                content=_multi_request_jsonl("gpt-4o-mini", BATCH_RL_REQUEST_LINES),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        return file

    def _generate_enqueued_key(
        self,
        client: BatchClient,
        resources: ResourceManager,
        *,
        limit: int,
        marker: str,
        rpm_limit: int | None = None,
    ) -> str:
        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[],
                rpm_limit=rpm_limit,
                user_id=f"e2e-batch-enq-{marker}-{unique_marker()}",
                metadata=KeyMetadata(batch_enqueued_token_limit=limit),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))
        return key

    @pytest.mark.covers(
        "quota_management.ratelimit.batch_enqueued_tokens.accepts_over_rpm",
        exercised_on=["batches"],
    )
    def test_enqueued_allowance_accepts_batch_over_key_rpm(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = self._generate_enqueued_key(
            client,
            resources,
            limit=BATCH_ENQUEUED_HEADROOM_TOKENS,
            marker="rpm",
            rpm_limit=BATCH_RL_RPM_LIMIT,
        )
        file = self._upload_batch_file(client, resources, key)

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)

        assert created.status_code != 429, (
            f"enqueued-token allowance must govern batch submission instead of the "
            f"key RPM ({BATCH_RL_RPM_LIMIT} < {BATCH_RL_REQUEST_LINES} rows); "
            f"got 429: {created.body[:400]}"
        )
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

    @pytest.mark.covers(
        "quota_management.ratelimit.batch_enqueued_tokens.blocks_when_exhausted",
        exercised_on=["batches"],
    )
    @pytest.mark.covers(
        "quota_management.ratelimit.batch_enqueued_tokens.refunds_on_cancel",
        exercised_on=["batches"],
    )
    def test_exhausted_allowance_blocks_until_cancel_refunds(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        sizing_key = self._generate_enqueued_key(
            client, resources, limit=1, marker="size"
        )
        sizing_file = self._upload_batch_file(client, resources, sizing_key)
        sized = client.create_batch(
            body=BatchCreateBody(input_file_id=sizing_file.id), key=sizing_key
        )
        assert sized.status_code == 429, (
            f"a 1-token allowance must reject any batch before it reaches the "
            f"provider, got {sized.status_code}: {sized.body[:400]}"
        )
        assert "batch enqueued token limit exceeded" in sized.body.lower(), (
            f"429 body must name the enqueued token limit, got: {sized.body[:400]}"
        )
        requires = _BATCH_REQUIRES_TOKENS.search(sized.body)
        assert requires is not None, (
            f"429 body must report the batch token requirement so callers can size "
            f"allowances, got: {sized.body[:400]}"
        )
        batch_tokens = int(requires.group(1))
        assert batch_tokens > 1

        key = self._generate_enqueued_key(
            client, resources, limit=batch_tokens + batch_tokens // 2, marker="refund"
        )
        file = self._upload_batch_file(client, resources, key)

        first = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(first)
        first_batch = BatchObject.model_validate_json(first.body)
        resources.defer(quietly(lambda: client.cancel_batch(first_batch.id, key=key)))

        blocked = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        assert blocked.status_code == 429, (
            f"second batch must not fit the remaining allowance while the first is "
            f"enqueued, got {blocked.status_code}: {blocked.body[:400]}"
        )
        assert "batch enqueued token limit exceeded" in blocked.body.lower(), (
            f"429 body must name the enqueued token limit, got: {blocked.body[:400]}"
        )

        cancelled = cancel_batch(client, first_batch.id, key=key, provider=None)
        assert cancelled.status in {"cancelling", "cancelled"}, (
            f"cancel must reach a cancel state for the refund to fire, "
            f"got {cancelled.status}"
        )

        retried = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        assert retried.status_code != 429, (
            f"cancelling the first batch must refund its reservation so the retry "
            f"fits the allowance, got 429: {retried.body[:400]}"
        )
        require_successful_call(retried)
        retry_batch = BatchObject.model_validate_json(retried.body)
        resources.defer(quietly(lambda: client.cancel_batch(retry_batch.id, key=key)))


ASSUME_ROLE_RAW_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _assume_role_params(role_arn: str, session_name: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=ASSUME_ROLE_RAW_MODEL,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
        s3_region_name="os.environ/AWS_REGION",
        s3_bucket_name="os.environ/AWS_BATCH_S3_BUCKET",
        s3_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        s3_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_batch_role_arn="os.environ/AWS_BATCH_ROLE_ARN",
        aws_role_name=role_arn,
        aws_session_name=session_name,
    )


class TestBedrockBatchAssumeRole:
    """Bedrock batch create under STS assume-role credentials.

    Provisions a bedrock batch deployment whose litellm_params carry
    aws_role_name / aws_session_name (the product path for role assumption) and
    runs the unified file-upload + batch-create lifecycle. Success means the
    proxy assumed the role and Bedrock accepted the job; a misconfigured role
    fails create with an AWS auth error rather than silently falling back to the
    ambient key.
    """

    @pytest.mark.covers(
        "llm.batches.bedrock.assume_role.nonstream.works",
        "llm.files.bedrock.upload.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_unified_batch_create_with_assume_role(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        role_arn = os.environ["AWS_ROLE_NAME"]
        session_name = f"e2e-batch-sts-{unique_marker()}"[:64]
        model_name = batch_model_name("bedrock-sts-batch")

        model_id = client.create_model(model_name, _assume_role_params(role_arn, session_name))
        resources.defer(lambda: client.delete_model(model_id))
        key = resources.key()

        file = unwrap(
            client.upload_file(
                content=render_jsonl(ASSUME_ROLE_RAW_MODEL),
                form=FileUploadForm(purpose="batch", target_model_names=model_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert_file_object(file, provider="bedrock")

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

        assert batch.id, f"assume-role create returned no batch id: {created.body[:200]}"
        assert is_managed_id(batch.id), (
            f"assume-role create via target_model_names must return a managed batch id, "
            f"got {batch.id!r}"
        )
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"assume-role batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)

        fetched = unwrap(client.retrieve_batch(batch.id, key=key))
        assert fetched.id == batch.id


GEMINI_FILES_RAW_MODEL = "gemini-2.5-flash"


class TestGeminiFiles:
    """Gemini Files API upload through the proxy (LIT-3382).

    gemini is a first-class FileCreateProvider. The test registers a gemini
    deployment, uploads a tiny batch-purpose JSONL with target_model_names
    routing, and asserts a FileObject comes back. Batch create for pure gemini
    (non-Vertex) is out of scope here; Vertex covers the Gemini batch job path in
    the main lifecycle matrix.
    """

    @pytest.mark.covers(
        "llm.files.gemini.upload.nonstream.works",
        exercised_on=["files"],
    )
    def test_gemini_file_upload(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        model_name = batch_model_name("gemini-files")
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(
                model=f"gemini/{GEMINI_FILES_RAW_MODEL}",
                api_key="os.environ/GEMINI_API_KEY",
            ),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = resources.key()

        file = unwrap(
            client.upload_file(
                content=render_jsonl(GEMINI_FILES_RAW_MODEL),
                form=FileUploadForm(purpose="batch", target_model_names=model_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert_file_object(file, provider="gemini")
        assert file.id, "gemini file upload returned no id"


def _vllm_params(api_base: str, api_key: str | None, model_id: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=f"hosted_vllm/{model_id}",
        api_base=api_base,
        api_key=api_key,
    )


class TestHostedVllmBatch:
    """hosted_vllm file upload + batch create (OpenAI-compatible path, LIT-3266).

    hosted_vllm is in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS, so /v1/files
    and /v1/batches route through the OpenAI handler against the deployment's
    api_base. Skipped for now: it needs a live vLLM (or OpenAI-compatible) server
    exposing the files/batches APIs (HOSTED_VLLM_API_BASE), which the e2e
    environment does not currently provision.
    """

    @pytest.mark.skip(
        reason="hosted_vllm batch/files needs a live vLLM server (HOSTED_VLLM_API_BASE) "
        "not provisioned in the e2e environment; re-enable when available (LIT-3266)"
    )
    @pytest.mark.covers(
        "llm.batches.hosted_vllm.basic.nonstream.works",
        "llm.files.hosted_vllm.upload.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_unified_file_and_batch_create(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        api_base = os.environ["HOSTED_VLLM_API_BASE"]
        api_key = (os.environ.get("HOSTED_VLLM_API_KEY") or "").strip() or None
        model_id = (
            os.environ.get("HOSTED_VLLM_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"
        ).strip()
        proxy_name = batch_model_name("hosted-vllm-batch")

        model_row_id = client.create_model(
            proxy_name, _vllm_params(api_base, api_key, model_id)
        )
        resources.defer(lambda: client.delete_model(model_row_id))
        key = resources.key()

        file = unwrap(
            client.upload_file(
                content=render_jsonl(model_id),
                form=FileUploadForm(purpose="batch", target_model_names=proxy_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert_file_object(file, provider="hosted_vllm")

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

        assert batch.id, f"hosted_vllm create returned no batch id: {created.body[:200]}"
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"hosted_vllm batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)


BATCH_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
FAILED_BATCH_POLL_SECONDS = 120.0
FAILED_BATCH_POLL_INTERVAL_SECONDS = 5.0

AZURE_BATCH_RAW_MODEL = next(p.raw_model for p in PROVIDERS if p.name == "azure")


def _mismatched_endpoint_jsonl(model: str) -> bytes:
    line = {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": model, "input": "ping"},
    }
    return (json.dumps(line) + "\n").encode()


def _poll_until_terminal(client: BatchClient, batch_id: str, key: str) -> BatchObject:
    deadline = time.monotonic() + FAILED_BATCH_POLL_SECONDS
    fetched = retrieve_batch(client, batch_id, key=key, provider=None)
    while fetched.status not in BATCH_TERMINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(FAILED_BATCH_POLL_INTERVAL_SECONDS)
        fetched = retrieve_batch(client, batch_id, key=key, provider=None)
    return fetched


class TestBatchFailurePaths:
    """Customer-facing failure contracts for /v1/batches.

    A malformed input file is rejected at upload with a 400 naming the bad
    content. A JSONL line whose url contradicts the batch endpoint is accepted
    at create (providers validate asynchronously) and drives the batch to
    "failed" with structured per-line errors, a null output_file_id, and a
    zero-cost spend row (LIT-4852: a failed batch must book $0, not crash cost
    tracking). Cancelling that already-failed batch returns a 409 naming the
    terminal status. A file id encoded for one deployment wins over a
    conflicting model param on create: the batch routes (and re-encodes) by the
    file's embedded model, pinning that precedence.
    """

    @pytest.mark.covers(
        "llm.batches.openai.malformed_jsonl.nonstream.works",
        exercised_on=["files"],
    )
    def test_malformed_jsonl_upload_rejected(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        result = client.upload_file(
            content=b"this is not json\n",
            form=FileUploadForm(purpose="batch"),
            model=OPENAI_BATCH_MODEL,
            key=resources.key(),
        )
        match result:
            case UnknownApiError(status_code=400, body=body):
                assert "json" in body.lower(), (
                    f"400 must name the malformed JSONL so users can fix the file, got: {body[:300]}"
                )
            case _:
                pytest.fail(f"malformed JSONL upload must be rejected with a 400, got: {result}")

    @pytest.mark.covers(
        "llm.batches.openai.jsonl_endpoint_mismatch.nonstream.works",
        "llm.batches.openai.cancel_terminal.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_endpoint_mismatch_fails_batch_and_cancel_conflicts(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        file = unwrap(
            client.upload_file(
                content=_mismatched_endpoint_jsonl("gpt-4o-mini"),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)

        fetched = _poll_until_terminal(client, batch.id, key)
        assert fetched.status == "failed", (
            f"endpoint-mismatched batch must fail, got {fetched.status!r}"
        )
        assert fetched.output_file_id is None, (
            f"failed batch must have no output file, got {fetched.output_file_id!r}"
        )
        assert fetched.errors is not None and fetched.errors.data, (
            "failed batch must surface structured errors so users can fix the JSONL"
        )
        first_error = fetched.errors.data[0]
        assert first_error.message, "batch error item has no message"
        assert first_error.code, "batch error item has no code"

        rows = client.proxy.poll_logs_for_request_id(f"{fetched.id}_batch_cost")
        assert rows, (
            f"failed batch {fetched.id} wrote no spend row; retrieve must book $0 (LIT-4852)"
        )
        assert all((row.spend or 0) == 0 for row in rows), (
            f"failed batch must cost $0, got {[(r.request_id, r.spend) for r in rows]}"
        )
        assert rows[0].call_type == "aretrieve_batch", (
            f"batch cost row call_type={rows[0].call_type!r}"
        )

        conflict = client.cancel_batch(batch.id, key=key)
        match conflict:
            case UnknownApiError(status_code=409, body=body):
                assert "failed" in body.lower(), (
                    f"409 must name the terminal status blocking the cancel, got: {body[:300]}"
                )
            case _:
                pytest.fail(f"cancel of a failed batch must return a 409 conflict, got: {conflict}")

    @pytest.mark.covers(
        "llm.batches.openai.foreign_file_id.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_foreign_encoded_file_id_routes_by_file_model(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        file = unwrap(
            client.upload_file(
                content=render_jsonl(AZURE_BATCH_RAW_MODEL),
                form=FileUploadForm(purpose="batch"),
                model=AZURE_BATCH_MODEL,
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert decoded_model_from_id(file.id) == AZURE_BATCH_MODEL, (
            f"upload did not encode the azure deployment into the file id: {file.id!r}"
        )

        created = client.create_batch(
            body=BatchCreateBody(input_file_id=file.id, model=OPENAI_BATCH_MODEL), key=key
        )
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

        assert decoded_model_from_id(batch.id) == AZURE_BATCH_MODEL, (
            "create with a foreign encoded file id must route by the file's embedded model, "
            f"but the batch id encodes {decoded_model_from_id(batch.id)!r} "
            f"(model param was {OPENAI_BATCH_MODEL!r})"
        )
        fetched = retrieve_batch(client, batch.id, key=key, provider=None)
        assert fetched.id == batch.id
        assert fetched.status, "retrieved foreign-file batch has no status"


class TestBatchSecondHop:
    """Two-proxy batch routing: a litellm_proxy deployment chained to the gateway
    itself (LIT-5347, PR #36240).

    The hop deployment's litellm_params point litellm_proxy/<inner model> at this
    gateway's own base URL with a freshly minted virtual key, so the unified
    upload and batch create traverse gateway -> gateway -> OpenAI. The regression
    this pins: target_model_names must be rewritten to the inner deployment on
    the second hop and the nested managed ids must round-trip retrieve.
    """

    @pytest.mark.covers(
        "llm.batches.openai.second_hop.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_unified_create_and_retrieve_via_chained_gateway(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        hop_name = batch_model_name("openai-batch-hop")
        model_id = client.create_model(
            hop_name,
            LiteLLMParamsBody(
                model=f"litellm_proxy/{OPENAI_BATCH_MODEL}",
                api_base=PROXY_BASE_URL,
                api_key=key,
            ),
        )
        resources.defer(lambda: client.delete_model(model_id))

        file = unwrap(
            client.upload_file(
                content=render_jsonl("gpt-4o-mini"),
                form=FileUploadForm(purpose="batch", target_model_names=hop_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert is_managed_id(file.id), (
            f"second-hop unified upload must return a managed file id, got {file.id!r}"
        )

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

        assert is_managed_id(batch.id), (
            f"second-hop create must return a managed batch id, got {batch.id!r}"
        )
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"second-hop batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)

        fetched = retrieve_batch(client, batch.id, key=key, provider=None)
        assert fetched.id == batch.id
        assert fetched.status, "second-hop retrieve returned no status"


class BatchOutputBody(BaseModel):
    choices: list[object] = []


class BatchOutputResponse(BaseModel):
    status_code: int | None = None
    body: BatchOutputBody | None = None


class BatchOutputLine(BaseModel):
    response: BatchOutputResponse


TERMINAL_MARKER_KEY = "litellm_e2e_suite"
TERMINAL_MARKER_VALUE = "batches-terminal-baton"
TERMINAL_POLL_SECONDS = 300.0
TERMINAL_POLL_INTERVAL_SECONDS = 10.0
TERMINAL_LIST_LIMIT = 100
TERMINAL_BAND_MIN_AGE_SECONDS = 25 * 3600
TERMINAL_BAND_MAX_AGE_SECONDS = 73 * 3600


def _marker_batches(client: BatchClient, key: str) -> list[BatchObject]:
    listed = unwrap(
        client.list_batches(key=key, model=OPENAI_BATCH_MODEL, limit=TERMINAL_LIST_LIMIT)
    )
    return [
        b
        for b in listed.data
        if (b.metadata or {}).get(TERMINAL_MARKER_KEY) == TERMINAL_MARKER_VALUE
    ]


def _await_completed_marker(
    client: BatchClient, key: str
) -> tuple[BatchObject | None, list[BatchObject]]:
    deadline = time.monotonic() + TERMINAL_POLL_SECONDS
    while True:
        markers = _marker_batches(client, key)
        completed = max(
            (b for b in markers if b.status == "completed"),
            key=lambda b: b.created_at or 0,
            default=None,
        )
        if completed is not None or time.monotonic() >= deadline:
            return completed, markers
        time.sleep(TERMINAL_POLL_INTERVAL_SECONDS)


def _assert_aged_markers_terminal(markers: list[BatchObject]) -> None:
    now = time.time()
    stuck = [
        b
        for b in markers
        if b.created_at is not None
        and TERMINAL_BAND_MIN_AGE_SECONDS <= now - b.created_at <= TERMINAL_BAND_MAX_AGE_SECONDS
        and b.status not in BATCH_TERMINAL_STATUSES
    ]
    assert not stuck, (
        "marker batches past their 24h completion window must be terminal; stuck: "
        f"{[(b.id, b.status, b.created_at) for b in stuck]}"
    )


class TestBatchTerminalState:
    """Terminal state + cost write-back via a cross-run marker baton.

    Each run submits a 1-line marker batch (stable metadata key/value plus a
    per-run field) and never cancels or deletes it: the marker is the baton the
    next run picks up. Polling is list-only for up to 5 minutes because a
    retrieve of a non-terminal batch books a $0 spend row whose request_id then
    blocks the real-cost row (skip_duplicates); the single retrieve happens only
    once a completed marker exists. The assertion target is the newest completed
    marker from ANY run, so on the 6h stage cadence the full assertions are
    deterministic from run 2 onward. On a cold start (no marker has ever
    completed within the poll budget) the test passes on the submission
    assertions alone: that is a documented vacuous pass, not a skip, and this
    run's marker becomes the next run's target. Markers aged past OpenAI's 24h
    completion window (25h-73h band, within the newest list page) must be
    terminal. The cost assertion is the LIT-5730 headline: retrieving a
    completed model-encoded batch must write a positive spend row keyed
    {batch_id}_batch_cost; before the fix the logging worker fetched the
    re-encoded output_file_id, 404d, and the row never landed.
    """

    @pytest.mark.covers(
        "llm.batches.openai.terminal_state.nonstream.works",
        "llm.batches.openai.terminal_state.nonstream.cost_logged",
        exercised_on=["batches", "files"],
    )
    def test_completed_batch_downloads_output_and_books_cost(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        file = unwrap(
            client.upload_file(
                content=render_jsonl("gpt-4o-mini"),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        created = client.create_batch(
            body=BatchCreateBody(
                input_file_id=file.id,
                metadata={
                    TERMINAL_MARKER_KEY: TERMINAL_MARKER_VALUE,
                    "run": unique_marker(),
                },
            ),
            key=key,
        )
        require_successful_call(created)
        submitted = BatchObject.model_validate_json(created.body)
        assert submitted.status in CREATED_BATCH_STATUSES, (
            f"marker batch has non-transitional status {submitted.status!r}"
        )
        assert (submitted.metadata or {}).get(TERMINAL_MARKER_KEY) == TERMINAL_MARKER_VALUE, (
            f"create dropped the marker metadata: {submitted.metadata!r}"
        )

        completed, markers = _await_completed_marker(client, key)
        _assert_aged_markers_terminal(markers)
        if completed is None:
            return

        fetched = retrieve_batch(client, completed.id, key=key, provider=None)
        assert fetched.status == "completed", (
            f"listed-completed marker retrieved as {fetched.status!r}"
        )
        assert fetched.output_file_id, "completed batch has no output_file_id"

        downloaded = client.proxy.transport.download(
            f"/v1/files/{fetched.output_file_id}/content",
            headers=client.proxy.transport.bearer(key),
        )
        assert downloaded.status_code == 200, (
            f"output content must be 200, got {downloaded.status_code}: {downloaded.body[:300]}"
        )
        first_line = BatchOutputLine.model_validate_json(downloaded.body.strip().splitlines()[0])
        assert first_line.response.status_code == 200, (
            f"batch output line reports failure: {downloaded.body[:400]}"
        )
        assert first_line.response.body is not None and first_line.response.body.choices, (
            "batch output line has no choices"
        )

        rows = client.proxy.poll_logs_for_request_id(
            f"{fetched.id}_batch_cost",
            predicate=lambda found: any((row.spend or 0) > 0 for row in found),
        )
        priced = [row for row in rows if (row.spend or 0) > 0]
        assert priced, (
            f"completed batch {fetched.id} wrote no positive-cost spend row under "
            f"request_id {fetched.id}_batch_cost; cost write-back is broken (LIT-5730)"
        )
        cost_row = priced[0]
        assert cost_row.call_type == "aretrieve_batch", (
            f"batch cost row call_type={cost_row.call_type!r}"
        )
        assert (cost_row.total_tokens or 0) > 0, (
            f"batch cost row has no token usage: {cost_row.total_tokens!r}"
        )
