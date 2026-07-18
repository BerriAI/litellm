"""Live e2e: batch create that exceeds a key's RPM maps to a structured 429.

The batch rate limiter reads the input file at submission time and rejects the
create when the file's request count would exceed the key's remaining RPM. The
product promise is not only the block itself but the OpenAI-compatible shape:
HTTP 429, a body that names the batch rate limit, and pacing headers so clients
can back off. Complements the LIT-3266 hygiene check (no orphan spend rows) by
asserting the error mapping when the limiter actually fires.
"""

from __future__ import annotations

import json

import pytest

from batch_client import BatchClient, BatchCreateBody
from capabilities import OPENAI_BATCH_MODEL
from e2e_config import unique_marker
from e2e_http import FileUploadForm, unwrap
from lifecycle import ResourceManager
from models import KeyGenerateBody
from test_batches_e2e import quietly

pytestmark = pytest.mark.e2e

REQUEST_LINES = 3
RPM_LIMIT = 1


def _multi_request_jsonl(model: str, n: int) -> bytes:
    lines: list[str] = []
    for i in range(n):
        lines.append(
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
        )
    return ("\n".join(lines) + "\n").encode()


class TestBatchRateLimitErrorMapping:
    @pytest.mark.covers(
        "quota_management.ratelimit.batch_rpm.blocks_over_limit",
        exercised_on=["batches"],
    )
    def test_batch_create_over_rpm_returns_mapped_429(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        user_id = f"e2e-batch-rl-map-{unique_marker()}"
        key = client.proxy.generate_key(
            KeyGenerateBody(models=[], rpm_limit=RPM_LIMIT, tpm_limit=1_000_000, user_id=user_id)
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        file = unwrap(
            client.upload_file(
                content=_multi_request_jsonl("gpt-4o-mini", REQUEST_LINES),
                form=FileUploadForm(purpose="batch"),
                model=OPENAI_BATCH_MODEL,
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)

        assert created.status_code == 429, (
            f"expected batch RPM 429 when file has {REQUEST_LINES} requests and "
            f"rpm_limit={RPM_LIMIT}, got {created.status_code}: {created.body[:400]}"
        )
        body_lower = created.body.lower()
        assert "batch rate limit exceeded" in body_lower, (
            f"429 body must name the batch rate limit so clients can branch on it; "
            f"got: {created.body[:400]}"
        )
        assert str(REQUEST_LINES) in created.body, (
            f"429 body should report the batch request count ({REQUEST_LINES}); "
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
