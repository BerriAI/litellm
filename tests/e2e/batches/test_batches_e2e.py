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
from typing import Callable

import pytest

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
    require_successful_call,
    unwrap,
)
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

CREATED_BATCH_STATUSES = {"validating", "in_progress", "finalizing"}
BATCH_CANCEL_DELAY_SECONDS = 2
BATCH_TERMINAL_BEFORE_CANCEL = {"failed", "cancelled", "expired"}


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


def assert_file_object(file: FileObject) -> None:
    assert file.object == "file", f"file.object={file.object!r}"
    assert file.purpose == "batch", f"file.purpose={file.purpose!r}"
    assert file.bytes is not None and file.bytes > 0, f"file.bytes={file.bytes!r}"
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
    cap: Capability, client: BatchClient, resources: ResourceManager
) -> None:
    key = resources.key()
    provider = op_provider(cap)

    file = unwrap(upload_for_scenario(client, cap, render_jsonl(cap.jsonl_model), key))
    resources.defer(
        quietly(lambda: client.delete_file(file.id, key=key, provider=provider))
    )
    assert_file_object(file)
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
        cancelled = unwrap(client.cancel_batch(batch.id, key=key, provider=provider))
        assert cancelled.id == batch.id
        assert cancelled.object == "batch"
        # Vertex cancel is async: the job may still show its pre-cancel status
        # briefly before transitioning to cancelling/cancelled.
        valid_post_cancel = {"cancelling", "cancelled"}
        if cap.provider == "vertex_ai":
            valid_post_cancel |= CREATED_BATCH_STATUSES
        assert cancelled.status in valid_post_cancel, (
            f"unexpected post-cancel status {cancelled.status!r}"
        )

    if cap.can_list:
        listed = unwrap(client.list_batches(key=key, provider=provider))
        # OpenAI includes object="list"; Azure provider list often omits the envelope field.
        if listed.object is not None:
            assert listed.object == "list", f"list envelope object={listed.object!r}"
        match = next((b for b in listed.data if b.id == batch.id), None)
        assert match is not None, "created batch absent from list"
        assert match.object == "batch"


def test_batch_key_model_access_denied(
    client: BatchClient, resources: ResourceManager
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
    client: BatchClient, resources: ResourceManager
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
    assert_file_object(file)

    deleted = unwrap(client.delete_file(file.id, key=key))
    assert deleted.id, "delete response has no id"
    assert deleted.object == "file", f"delete object={deleted.object!r}"
    assert deleted.deleted is True, "file was not reported deleted"


def test_anthropic_batch_retrieve(client: BatchClient, scoped_key: str) -> None:
    batch_id = os.environ.get("ANTHROPIC_BATCH_ID")
    if not batch_id:
        pytest.skip(
            "set ANTHROPIC_BATCH_ID to a real anthropic batch id to exercise retrieve"
        )
    fetched = unwrap(
        client.retrieve_batch(batch_id, key=scoped_key, provider="anthropic")
    )
    assert fetched.id == batch_id
    assert fetched.status
