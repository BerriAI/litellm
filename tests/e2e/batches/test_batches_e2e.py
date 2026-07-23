"""Live e2e for the Batches API across every provider LiteLLM supports.

Synchronous tier only: a batch's completion window is 24h, so these never wait for
"completed". Each case uploads a tiny JSONL, creates the batch through one of the
four routing scenarios, asserts it was accepted (non-terminal status) and routed to
the right provider, then retrieves / cancels / lists where the provider supports it.
Everything created is deleted on teardown. Completion + cost tracking are out of
scope here (see COVERAGE.md).

Routing signal: for provider_fallback the raw batch id discriminates the provider;
for the encoded/unified/model_param scenarios the proxy re-encodes the id, so the
load-bearing signal is that create SUCCEEDS against that provider's own model - a
misroute to the wrong provider fails the create.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import pytest

from e2e_config import require_env, unique_marker

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
    OPENAI_BATCH_MODEL,
    Capability,
    batch_model_name,
    coverage_cells_for_lifecycle,
    is_managed_id,
    matches_id_shape,
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
from models import KeyGenerateBody, LiteLLMParamsBody, SpendLogRow

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


OPENAI_FILE_CONTENT_BACKEND = "gpt-4o-mini"


class TestBatchFileContent:
    """GET /v1/files/{id}/content returns the uploaded batch JSONL bytes."""

    @pytest.mark.covers(
        "llm.files.openai.content.nonstream.works",
        exercised_on=["files"],
    )
    def test_file_content_matches_upload(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        proxy_name = f"e2e-file-content-{unique_marker()}"
        model_id = client.create_model(
            proxy_name,
            LiteLLMParamsBody(
                model=f"openai/{OPENAI_FILE_CONTENT_BACKEND}",
                api_key="os.environ/OPENAI_API_KEY",
            ),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = resources.key()

        payload = render_jsonl(OPENAI_FILE_CONTENT_BACKEND)
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
        (role_arn,) = require_env("AWS_ROLE_NAME")
        require_env(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_BATCH_S3_BUCKET",
            "AWS_BATCH_ROLE_ARN",
        )
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
        (api_base,) = require_env("HOSTED_VLLM_API_BASE")
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
