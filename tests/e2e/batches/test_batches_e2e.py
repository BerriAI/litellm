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
import time
from typing import Callable

import pytest

from e2e_config import unique_marker

from batch_client import (
    BatchClient,
    BatchCreateBody,
    BatchObject,
    FileObject,
    is_model_access_denied,
    is_result_access_denied,
)
from capabilities import (
    BATCH_ID_SHAPE,
    CAPABILITIES,
    FILE_ID_SHAPE,
    Capability,
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
from models import KeyGenerateBody, SpendLogRow, SpendLogsParams

pytestmark = pytest.mark.e2e

CREATED_BATCH_STATUSES = {"validating", "in_progress", "finalizing"}
BATCH_CANCEL_DELAY_SECONDS = 2
BATCH_TERMINAL_BEFORE_CANCEL = {"failed", "cancelled", "expired"}
BATCH_CANCEL_RETRIES = 3


def cancel_batch(
    client: BatchClient, batch_id: str, *, key: str, provider: str | None
) -> BatchObject:
    last = client.cancel_batch(batch_id, key=key, provider=provider)
    for _ in range(BATCH_CANCEL_RETRIES - 1):
        match last:
            case Success(data=data):
                return data
            case UnknownApiError(status_code=500):
                time.sleep(1)
                last = client.cancel_batch(batch_id, key=key, provider=provider)
            case _:
                break
    return unwrap(last)


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


@pytest.mark.parametrize("cap", CAPABILITIES, ids=[c.id for c in CAPABILITIES])
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

    created = create_for_scenario(client, cap, file.id, key)
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

    fetched = unwrap(client.retrieve_batch(batch.id, key=key, provider=provider))
    assert_batch_object(fetched)
    assert fetched.id == batch.id
    assert (
        fetched.input_file_id == batch.input_file_id
    ), "retrieve changed input_file_id"
    assert fetched.status, "retrieved batch has no status"

    if cap.can_cancel:
        time.sleep(BATCH_CANCEL_DELAY_SECONDS)
        pre_cancel = unwrap(client.retrieve_batch(batch.id, key=key, provider=provider))
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
        valid_post_cancel = {"cancelling", "cancelled"}
        if cap.provider == "vertex_ai":
            valid_post_cancel |= CREATED_BATCH_STATUSES
        assert cancelled.status in valid_post_cancel, (
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


def test_batch_key_model_access_denied(
    client: BatchClient, resources: ResourceManager, batch_deployments: None
) -> None:
    key = resources.key(models=["openai-batch"])

    denied_upload = client.upload_file(
        content=render_jsonl("azure-batch"),
        form=FileUploadForm(purpose="batch"),
        model="azure-batch",
        key=key,
    )
    assert is_result_access_denied(
        denied_upload
    ), f"restricted key uploaded a file for a disallowed model: {denied_upload}"

    raw_file = unwrap(
        client.upload_file(
            content=render_jsonl("openai-batch"),
            form=FileUploadForm(purpose="batch"),
            key=key,
            provider="openai",
        )
    ).id
    resources.defer(
        quietly(lambda: client.delete_file(raw_file, key=key, provider="openai"))
    )

    denied_create = client.create_batch(
        body=BatchCreateBody(input_file_id=raw_file, model="azure-batch"), key=key
    )
    assert is_model_access_denied(
        denied_create
    ), f"restricted key created a batch for a disallowed model (status {denied_create.status_code})"


def test_file_upload_and_delete_outputs(
    client: BatchClient, resources: ResourceManager, batch_deployments: None
) -> None:
    key = resources.key()
    file = unwrap(
        client.upload_file(
            content=render_jsonl("openai-batch"),
            form=FileUploadForm(purpose="batch"),
            model="openai-batch",
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
    """
    user_id = f"e2e-batch-rl-{unique_marker()}"
    key = client.gateway.generate_key(
        KeyGenerateBody(models=[], tpm_limit=1_000_000, rpm_limit=1_000, user_id=user_id)
    )
    resources.defer(lambda: client.gateway.delete_key(key))

    before = frozenset(
        row.request_id for row in unattributed_rows(client.gateway.spend_logs(SpendLogsParams()))
    )

    file = unwrap(
        client.upload_file(
            content=render_jsonl("gpt-4o-mini"),
            form=FileUploadForm(purpose="batch"),
            model="openai-batch",
            key=key,
        )
    )
    resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

    created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
    require_successful_call(created)
    batch = BatchObject.model_validate_json(created.body)
    resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

    _ = client.gateway.poll_logs_for_key(key, min_rows=1)

    new_orphans = [
        row
        for row in unattributed_rows(client.gateway.spend_logs(SpendLogsParams()))
        if row.request_id not in before
    ]
    assert not new_orphans, (
        "batch create on a rate-limited key left an unattributed spend row "
        f"(LIT-3266); rows={[(r.request_id, r.call_type, r.model) for r in new_orphans]}"
    )
