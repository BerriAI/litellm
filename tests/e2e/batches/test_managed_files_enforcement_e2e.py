"""Live e2e pins for litellm_settings.require_managed_files enforcement.

require_managed_files is a boot-time module global, so these tests need a proxy
whose config enables it. The main ephemeral stack can never run with it on: the
flag would 400 every files_settings-routed upload in the rest of the suite. The
PR gate instead reconfigures the same stack sequentially after the main run and
executes only this file with E2E_MANAGED_FILES_STACK set; without that env every
test here is deselected (see conftest.py, mirroring the weekly marker).

Pins: an upload without target_model_names is rejected 400, an upload that also
carries a model param is rejected 400, a raw provider file id is rejected 400 on
retrieve, and another user's managed unified file id is denied 403 while the
owning user still retrieves it.
"""

from __future__ import annotations

import json
from typing import Iterator

import pytest

from batch_client import BatchClient, FileObject
from capabilities import batch_model_name, is_managed_id, openai_batch_params
from e2e_config import unique_marker
from e2e_http import FileUploadForm, Result, UnknownApiError, unwrap
from lifecycle import ResourceManager

pytestmark = [pytest.mark.e2e, pytest.mark.managed_files]

UPLOAD_ROW = "llm.files.openai.require_managed_files_upload.nonstream.works"
ISOLATION_ROW = "llm.files.openai.require_managed_files_isolation.nonstream.works"


def batch_jsonl(model: str) -> bytes:
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


def expect_api_error(result: Result[FileObject], status: int, needle: str) -> None:
    match result:
        case UnknownApiError(status_code=code, body=body) if code == status:
            assert needle in body, f"expected {needle!r} in HTTP {status} body: {body[:300]}"
        case _:
            raise AssertionError(f"expected HTTP {status} containing {needle!r}, got: {result}")


@pytest.fixture(scope="module")
def managed_model(client: BatchClient) -> Iterator[str]:
    model_name = batch_model_name("managed-files-openai")
    model_id = client.create_model(model_name, openai_batch_params())
    yield model_name
    client.delete_model(model_id)


@pytest.mark.covers(UPLOAD_ROW)
def test_upload_without_target_model_names_rejected(
    client: BatchClient, scoped_key: str, managed_model: str
) -> None:
    result = client.upload_file(
        content=batch_jsonl(managed_model),
        form=FileUploadForm(purpose="batch"),
        key=scoped_key,
    )
    expect_api_error(result, 400, "target_model_names is required")


@pytest.mark.covers(UPLOAD_ROW)
def test_upload_with_model_param_rejected(
    client: BatchClient, scoped_key: str, managed_model: str
) -> None:
    result = client.upload_file(
        content=batch_jsonl(managed_model),
        form=FileUploadForm(purpose="batch", target_model_names=managed_model),
        model=managed_model,
        key=scoped_key,
    )
    expect_api_error(result, 400, "model is not allowed")


@pytest.mark.covers(ISOLATION_ROW)
def test_raw_provider_file_id_rejected(client: BatchClient, scoped_key: str) -> None:
    result = client.retrieve_file("file-e2e-raw-provider-id", key=scoped_key)
    expect_api_error(result, 400, "Raw provider file ids cannot be used")


@pytest.mark.covers(ISOLATION_ROW)
def test_cross_user_managed_id_denied_owner_allowed(
    client: BatchClient, resources: ResourceManager, managed_model: str
) -> None:
    run = unique_marker()
    owner_key = resources.key(user_id=f"managed-files-owner-{run}")
    other_key = resources.key(user_id=f"managed-files-other-{run}")

    uploaded = unwrap(
        client.upload_file(
            content=batch_jsonl(managed_model),
            form=FileUploadForm(purpose="batch", target_model_names=managed_model),
            key=owner_key,
        )
    )
    resources.defer(lambda: client.delete_file(uploaded.id, key=owner_key))
    assert is_managed_id(uploaded.id), f"expected a managed unified file id, got {uploaded.id}"

    denied = client.retrieve_file(uploaded.id, key=other_key)
    expect_api_error(denied, 403, "does not have access to this managed file")

    retrieved = unwrap(client.retrieve_file(uploaded.id, key=owner_key))
    assert retrieved.id == uploaded.id
