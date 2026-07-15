"""Large-file uploads on /v1/files and batch rate limiting.

Two behaviors live here that share the same live batches infrastructure:

1. Large-file upload across every batch provider. The proxy spools the multipart
   body to a starlette SpooledTemporaryFile and streams from that handle when
   purpose=="batch"; the failure mode this test catches is a regression that
   would re-read the whole file into memory (OOM) or truncate the upstream
   payload. The test asserts the round-trip: the file id the proxy returns
   is usable to CREATE a batch with the provider's own model, which proves the
   upstream received the file (a truncated / not-uploaded file fails create).

   Default runs use a small file so CI stays cheap. Set
   LITELLM_E2E_LARGE_UPLOAD_MB=500 to exercise the 500MB path against real
   providers (real $$$).

2. A team key with a tight rpm limit. The batch rate limiter is deliberate:
   at create time it downloads the input JSONL, counts requests + tokens, and
   reserves that many against the caller's rpm/tpm counters BEFORE the batch
   is submitted upstream (see litellm/proxy/hooks/batch_rate_limiter.py). Why
   this exists: a single POST /v1/batches with, say, 5000 rows spawns 5000
   upstream chat calls once the provider drains the batch. Without pre-counting
   the whole file, a caller with rpm_limit=10 could smuggle 5000 requests
   through one HTTP call and skip the per-request limiter entirely. This test
   pins that behavior: an rpm-limited team key trying to create a batch whose
   row count exceeds the limit gets a 429.
"""

from __future__ import annotations

import json
import os

import pytest

from batch_client import BatchClient, BatchCreateBody, BatchObject, FileObject
from capabilities import CAPABILITIES, Capability, matches_id_shape, FILE_ID_SHAPE
from e2e_http import (
    FileUploadForm,
    NoBody,
    Result,
    require_successful_call,
    unwrap,
)
from lifecycle import ResourceManager
from models import KeyGenerateBody, TeamDeleteBody, TeamNewBody, TeamNewResponse
from test_batches_e2e import (
    assert_batch_object,
    assert_file_object,
    create_for_scenario,
    op_provider,
    quietly,
    upload_for_scenario,
    CREATED_BATCH_STATUSES,
)

pytestmark = pytest.mark.e2e


DEFAULT_UPLOAD_MB = 4
LARGE_UPLOAD_MB_ENV = "LITELLM_E2E_LARGE_UPLOAD_MB"


def upload_size_mb() -> int:
    raw = os.environ.get(LARGE_UPLOAD_MB_ENV, "").strip()
    if not raw:
        return DEFAULT_UPLOAD_MB
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{LARGE_UPLOAD_MB_ENV}={raw!r} must be an integer number of megabytes"
        ) from exc
    if parsed < 1:
        raise ValueError(f"{LARGE_UPLOAD_MB_ENV}={parsed} must be >= 1")
    return parsed


def render_jsonl_at_least(model: str, min_bytes: int) -> bytes:
    """Build a well-formed batch JSONL whose total size is >= min_bytes.

    Every line is a valid /v1/chat/completions request so the upstream provider
    will accept it if it received the whole file; the proxy never parses lines
    during upload, so this doubles as a smoke test that the multipart stream
    was not truncated.
    """
    template = {
        "custom_id": "req-000000",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        },
    }
    one_line = (json.dumps(template) + "\n").encode()
    per_line = len(one_line)
    line_count = max(1, (min_bytes + per_line - 1) // per_line)
    lines: list[bytes] = []
    for index in range(line_count):
        row = dict(template)
        row["custom_id"] = f"req-{index:09d}"
        lines.append((json.dumps(row) + "\n").encode())
    return b"".join(lines)


@pytest.mark.parametrize("cap", CAPABILITIES, ids=[c.id for c in CAPABILITIES])
def test_large_batch_file_upload_reaches_upstream(
    cap: Capability,
    client: BatchClient,
    resources: ResourceManager,
    batch_deployments: None,
) -> None:
    """A ~size_mb-large JSONL uploaded via /v1/files is accepted, round-trips a
    correctly shaped id, and the returned id is usable to create a batch with
    the provider's own model. Create success is the load-bearing signal that
    the upstream actually received the file; a truncated or memory-only
    upload would fail create (bad file id / empty body upstream)."""
    size_mb = upload_size_mb()
    min_bytes = size_mb * 1024 * 1024
    content = render_jsonl_at_least(cap.jsonl_model, min_bytes)
    key = resources.key()
    provider_hint = op_provider(cap)

    upload_result: Result[FileObject] = upload_for_scenario(client, cap, content, key)
    file = unwrap(upload_result)
    resources.defer(
        quietly(lambda: client.delete_file(file.id, key=key, provider=provider_hint))
    )
    assert_file_object(file, provider=cap.provider)
    assert matches_id_shape(FILE_ID_SHAPE[cap.scenario], file.id), (
        f"{cap.id}: file id {file.id!r} has wrong shape for scenario "
        f"{cap.scenario!r}"
    )
    if cap.provider != "bedrock" and file.bytes is not None:
        assert file.bytes >= len(content), (
            f"proxy reported bytes={file.bytes} but uploaded {len(content)}; "
            "upstream likely received a truncated file"
        )

    created = create_for_scenario(client, cap, file.id, key)
    require_successful_call(created)
    batch = BatchObject.model_validate_json(created.body)
    resources.defer(
        quietly(lambda: client.cancel_batch(batch.id, key=key, provider=provider_hint))
    )
    assert batch.id, "create returned no batch id after large-file upload"
    assert batch.status in CREATED_BATCH_STATUSES, (
        f"large-file batch went straight to {batch.status!r}; "
        "provider likely rejected the upload"
    )
    assert_batch_object(batch)


def create_team(client: BatchClient, alias: str) -> str:
    return unwrap(
        client.gateway.transport.post(
            "/team/new",
            headers=client.gateway.transport.master,
            json=TeamNewBody(team_alias=alias),
            response_type=TeamNewResponse,
        )
    ).team_id


def delete_team(client: BatchClient, team_id: str) -> None:
    _ = client.gateway.transport.post(
        "/team/delete",
        headers=client.gateway.transport.master,
        json=TeamDeleteBody(team_ids=[team_id]),
        response_type=NoBody,
    )


def test_team_key_rpm_limit_blocks_oversized_batch(
    client: BatchClient,
    resources: ResourceManager,
    batch_deployments: None,
) -> None:
    """A team key with rpm_limit=N must reject a batch whose input JSONL
    contains >N requests. The proxy's batch rate limiter (batch_rate_limiter.py
    async_pre_call_hook) reads the input file at submission time, counts rows,
    and rejects with 429 rather than letting one HTTP call fan out into
    thousands of upstream requests that bypass the per-request rpm limiter.

    Scenario: an internal user is issued a team key with tight rpm/tpm;
    their large batch's chunk count exceeds the limit; the proxy must 429
    on create, not accept the batch and let the provider drain it.
    """
    team_id = create_team(client, alias=f"e2e-batch-rl-{os.urandom(4).hex()}")
    resources.defer(lambda: delete_team(client, team_id))

    rpm_limit = 5
    request_count = rpm_limit * 20
    key = client.gateway.generate_key(
        KeyGenerateBody(
            models=["openai-batch"],
            team_id=team_id,
            rpm_limit=rpm_limit,
            tpm_limit=10_000,
            user_id=f"e2e-batch-rl-user-{os.urandom(4).hex()}",
        )
    )
    resources.defer(lambda: client.gateway.delete_key(key))

    def make_line(index: int) -> str:
        row: dict[str, object] = {
            "custom_id": f"req-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
            },
        }
        return json.dumps(row)

    content = ("\n".join(make_line(i) for i in range(request_count)) + "\n").encode()

    file = unwrap(
        client.upload_file(
            content=content,
            form=FileUploadForm(purpose="batch"),
            model="openai-batch",
            key=key,
        )
    )
    resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

    created = client.create_batch(
        body=BatchCreateBody(input_file_id=file.id), key=key
    )
    assert created.status_code == 429, (
        f"expected 429 from batch rate limiter for {request_count} requests on "
        f"rpm_limit={rpm_limit}; got status={created.status_code}, "
        f"body={created.body[:400]}"
    )
    assert (
        "batch" in created.body.lower()
        and ("rate limit" in created.body.lower() or "rpm" in created.body.lower())
    ), (
        "429 body does not look like the batch rate limiter's message; "
        f"body={created.body[:400]}"
    )
