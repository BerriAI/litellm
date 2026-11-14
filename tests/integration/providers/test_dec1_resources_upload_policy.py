"""Contract: proxy-wide upload policy set by the operator in `config.yaml`.

Two independent policies are pinned here, each on its own owned proxy.

Upload limits in `general_settings`: `max_file_size_mb` caps every upload, `max_batch_file_size_mb` caps
`purpose=batch` uploads, `allowed_file_extensions` is an allowlist and `blocked_file_extensions` a denylist
(both compare the lower-cased extension including its dot). A refused upload is an OpenAI-shaped 413 or 400
with `param: file`, an exact message, and no upstream request.

`litellm_settings.require_managed_files: true`: every upload must go through managed files (it must name
`target_model_names` and must not name `model`), and every route that takes a file or batch id refuses a raw
provider id with 400. This is the switch that makes ownership checks unavoidable, so a rewrite that forgets
one route reopens cross-tenant access by raw id.

Assumed: `dec1_resources_support.booted_rig` merges the given settings into the generated config.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from integration.providers.dec1_resources_support import (
    OPENAI_MODEL,
    Rig,
    batch_line,
    booted_rig,
    error_of,
    jsonl,
    unique,
)

_MB: Final = 1024 * 1024
_NOT_FORWARDED: Final = "The file was not forwarded to the provider."
_LINE: Final = jsonl(batch_line("gpt-4o-mini"))
_RAW_FILE: Final = (
    "Raw provider file ids cannot be used when require_managed_files is enabled in litellm_settings. "
    "Use the LiteLLM managed file id returned when the file was created."
)
_RAW_BATCH: Final = (
    "Raw provider batch ids cannot be used when require_managed_files is enabled in litellm_settings. "
    "Use the LiteLLM managed batch id returned when the batch was created."
)


@pytest.fixture(scope="module")
def limited(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(
        tmp_path_factory.mktemp("dec1-resources-upload-limits"),
        general_settings={
            "max_file_size_mb": 2,
            "max_batch_file_size_mb": 1,
            "allowed_file_extensions": [".jsonl", ".txt", ".exe"],
            "blocked_file_extensions": [".EXE"],
        },
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def managed_only(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(
        tmp_path_factory.mktemp("dec1-resources-require-managed"),
        litellm_settings={"require_managed_files": True},
    ) as booted:
        yield booted


def _refused(rig: Rig, response: httpx.Response, status: int, message: str, param: str | None = "file") -> None:
    assert response.status_code == status, response.text
    error: Final = error_of(response)
    assert error["message"] == message
    assert error["param"] == param
    assert rig.drained() == ()


def _batch_file_of(size: int) -> bytes:
    """One valid batch line padded to exactly `size` bytes."""
    empty: Final = jsonl(batch_line("gpt-4o-mini", content=""))
    return jsonl(batch_line("gpt-4o-mini", content="x" * (size - len(empty))))


def test_upload_one_byte_over_max_file_size_mb_is_413(limited: Rig) -> None:
    """`max_file_size_mb` is 2 and a `purpose=assistants` upload is 2 MB plus one byte.

    The proxy answers 413, `param: file`, message `Uploaded file exceeds the configured max_file_size_mb of 2
    MB (read stopped at 2.0 MB). ...`. The read stops one byte past the limit, so the reported size is the
    limit, not the true size. Nothing is sent upstream.
    """
    limited.wire.drain()
    _refused(
        limited,
        limited.upload("big.txt", b"a" * (2 * _MB + 1), purpose="assistants", content_type="text/plain"),
        413,
        f"Uploaded file exceeds the configured max_file_size_mb of 2 MB (read stopped at 2.0 MB). {_NOT_FORWARDED}",
    )


def test_upload_exactly_at_max_file_size_mb_is_accepted(limited: Rig) -> None:
    """`max_file_size_mb` is 2 and a `purpose=assistants` upload is exactly 2 MB.

    The limit is inclusive, so the upload is accepted with 200 and reaches the upstream once.
    """
    limited.wire.drain()
    response: Final = limited.upload(
        f"{unique('exact')}.txt", b"a" * (2 * _MB), purpose="assistants", content_type="text/plain"
    )
    assert response.status_code == 200, response.text
    assert len(limited.drained()) == 1


def test_batch_upload_over_max_batch_file_size_mb_is_413_with_the_real_size(limited: Rig) -> None:
    """`max_batch_file_size_mb` is 1 (below the general 2 MB cap) and a valid batch file is 1.5 MB.

    The proxy answers 413, `param: file`, message `Batch input file is 1.5 MB, which exceeds the configured
    max_batch_file_size_mb of 1 MB. ...`. A non-batch upload of the same size would pass, which the next test
    shows.
    """
    limited.wire.drain()
    _refused(
        limited,
        limited.upload("big.jsonl", _batch_file_of(_MB + _MB // 2)),
        413,
        "Batch input file is 1.5 MB, which exceeds the configured max_batch_file_size_mb of 1 MB. "
        f"{_NOT_FORWARDED}",
    )


def test_max_batch_file_size_mb_does_not_apply_to_other_purposes(limited: Rig) -> None:
    """With the same limits, a 1.5 MB file is uploaded with `purpose=assistants`.

    Only the general 2 MB cap applies, so the upload is accepted with 200.
    """
    response: Final = limited.upload(
        f"{unique('mid')}.txt", b"a" * (_MB + _MB // 2), purpose="assistants", content_type="text/plain"
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("filename", "subject"),
    [("report.pdf", "File extension '.pdf'"), ("README", "A file without an extension")],
    ids=["unlisted-extension", "no-extension"],
)
def test_upload_outside_allowed_file_extensions_is_400(limited: Rig, filename: str, subject: str) -> None:
    """`allowed_file_extensions` lists `.jsonl`, `.txt` and `.exe`; a `.pdf` file and a file with no extension
    are uploaded.

    Each is refused with 400, `param: file`, message `<subject> is not in this proxy's
    allowed_file_extensions setting. The file was not forwarded to the provider.`
    """
    limited.wire.drain()
    _refused(
        limited,
        limited.upload(filename, b"content", purpose="assistants", content_type="application/octet-stream"),
        400,
        f"{subject} is not in this proxy's allowed_file_extensions setting. {_NOT_FORWARDED}",
    )


def test_upload_with_a_blocked_extension_is_400_even_when_also_allowed(limited: Rig) -> None:
    """`.exe` is in both `allowed_file_extensions` and (as `.EXE`) `blocked_file_extensions`; `Tool.Exe` is
    uploaded.

    The denylist wins and matching ignores case: 400, `param: file`, message `File extension '.exe' is
    blocked by this proxy's blocked_file_extensions setting. ...`.
    """
    limited.wire.drain()
    _refused(
        limited,
        limited.upload("Tool.Exe", b"MZ", purpose="assistants", content_type="application/octet-stream"),
        400,
        f"File extension '.exe' is blocked by this proxy's blocked_file_extensions setting. {_NOT_FORWARDED}",
    )


def test_require_managed_files_refuses_an_upload_without_target_model_names(managed_only: Rig) -> None:
    """`require_managed_files` is on and a file is uploaded with only `purpose` and `file`.

    The proxy answers 400 with the exact message telling the caller to send `target_model_names`, and the
    provider is not called.
    """
    managed_only.wire.drain()
    _refused(
        managed_only,
        managed_only.upload("plain.jsonl", _LINE),
        400,
        "target_model_names is required when require_managed_files is enabled in litellm_settings. "
        "Provide one or more model aliases via the target_model_names form field "
        "(e.g. target_model_names=my-model-alias).",
        None,
    )


def test_require_managed_files_refuses_an_upload_that_also_names_model(managed_only: Rig) -> None:
    """`require_managed_files` is on and an upload names both `target_model_names` and `model`.

    `model` would route around the managed-files hook, so the proxy answers 400 with the exact message saying
    `model is not allowed`, and the provider is not called.
    """
    managed_only.wire.drain()
    _refused(
        managed_only,
        managed_only.upload("both.jsonl", _LINE, target_model_names=OPENAI_MODEL, model=OPENAI_MODEL),
        400,
        "model is not allowed when require_managed_files is enabled in litellm_settings. "
        "Uploads must go through managed files using target_model_names instead of the model parameter.",
        None,
    )


def test_require_managed_files_accepts_a_managed_upload(managed_only: Rig) -> None:
    """`require_managed_files` is on and an upload names `target_model_names` only.

    The upload is accepted with 200, reaches the upstream once, and the id returned is not the raw provider
    id (it does not start with `file-<stem>`).
    """
    stem: Final = unique("managed-ok")
    managed_only.wire.drain()
    file_id: Final = managed_only.uploaded_id(f"{stem}.jsonl", _LINE, target_model_names=OPENAI_MODEL)
    assert file_id != f"file-{stem}"
    assert managed_only.trail() == [("POST", "/v1/files")]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/files/file-raw-provider-id"),
        ("GET", "/v1/files/file-raw-provider-id/content"),
        ("DELETE", "/v1/files/file-raw-provider-id"),
    ],
    ids=["retrieve", "content", "delete"],
)
def test_require_managed_files_refuses_a_raw_file_id_on_every_file_route(
    managed_only: Rig, method: str, path: str
) -> None:
    """`require_managed_files` is on and the master key reads, downloads or deletes a raw provider file id.

    Even an admin gets 400 with the exact "Raw provider file ids cannot be used" message, and the provider is
    not called. Ownership is recorded only for managed ids, so a raw id has no tenant check to pass.
    """
    managed_only.wire.drain()
    _refused(managed_only, managed_only.proxy.request(method, path), 400, _RAW_FILE, None)


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/v1/batches/batch_raw-provider-id"), ("POST", "/v1/batches/batch_raw-provider-id/cancel")],
    ids=["retrieve", "cancel"],
)
def test_require_managed_files_refuses_a_raw_batch_id(managed_only: Rig, method: str, path: str) -> None:
    """`require_managed_files` is on and the master key retrieves or cancels a raw provider batch id.

    The proxy answers 400 with the exact "Raw provider batch ids cannot be used" message and the provider is
    not called.
    """
    managed_only.wire.drain()
    _refused(managed_only, managed_only.proxy.request(method, path), 400, _RAW_BATCH, None)


def test_require_managed_files_refuses_a_batch_created_from_a_raw_input_file_id(managed_only: Rig) -> None:
    """`require_managed_files` is on and a batch is created with `input_file_id` set to a raw provider id.

    The proxy answers 400 with the raw file id message, before reading the input file or creating anything
    upstream.
    """
    managed_only.wire.drain()
    response: Final = managed_only.proxy.request(
        "POST",
        "/v1/batches",
        {"input_file_id": "file-raw-provider-id", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
    )
    _refused(managed_only, response, 400, _RAW_FILE, None)
