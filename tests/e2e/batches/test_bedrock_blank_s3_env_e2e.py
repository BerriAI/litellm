"""Live e2e pin for Bedrock batch create with blank AWS_S3_* env vars.

Owns its own file (not test_batches_e2e.py) so the PR changed-file e2e gate
stays a single tiny file: this class boots its own gateway with
AWS_S3_ENCRYPTION_KEY_ID and AWS_S3_BUCKET_OWNER exported empty, then runs the
unified target_model_names upload + batch create lifecycle against real Bedrock.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from batch_cleanup import cleanup_batch, cleanup_file
from batch_client import BatchClient, BatchCreateBody, BatchObject, FileObject
from bedrock_env_gateway import BedrockEnvGateway
from capabilities import is_managed_id
from e2e_http import FileUploadForm, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import KeyGenerateBody

pytestmark = pytest.mark.e2e

CREATED_BATCH_STATUSES = {"validating", "in_progress", "finalizing"}
BLANK_S3_RAW_MODEL: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


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


def assert_file_object(file: FileObject, *, provider: str) -> None:
    assert file.object == "file", f"file.object={file.object!r}"
    assert file.purpose == "batch", f"file.purpose={file.purpose!r}"
    assert file.bytes is not None, f"file.bytes={file.bytes!r}"
    if provider != "bedrock":
        assert file.bytes > 0, f"file.bytes={file.bytes!r}"
    assert file.status, "file.status missing"
    assert file.created_at is not None and file.created_at > 0, "file.created_at missing"


def assert_batch_object(batch: BatchObject) -> None:
    assert batch.object == "batch", f"batch.object={batch.object!r}"
    if batch.endpoint:
        assert batch.endpoint == "/v1/chat/completions", f"batch.endpoint={batch.endpoint!r}"
    assert batch.completion_window == "24h", f"window={batch.completion_window!r}"
    assert batch.input_file_id, "batch.input_file_id missing"
    assert batch.created_at is not None and batch.created_at > 0, "batch.created_at missing"


class TestBedrockBatchBlankS3EnvVars:
    """Bedrock batch create with AWS_S3_* env vars exported but blank.

    Regression: a blank AWS_S3_ENCRYPTION_KEY_ID or AWS_S3_BUCKET_OWNER env var
    resolved to "" and was serialized into the create-job request, which Bedrock
    rejects. The owned gateway exports both vars empty, so the unified lifecycle
    only passes when blank is treated as unset.
    """

    @pytest.mark.covers(
        "llm.batches.bedrock.blank_s3_env.nonstream.works",
        "llm.files.bedrock.upload.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_unified_batch_create_ignores_blank_s3_env_vars(self, resources: ResourceManager) -> None:
        gateway: Final = BedrockEnvGateway.start()
        resources.defer(gateway.stop)
        client: Final = BatchClient(proxy=gateway.proxy)

        key: Final = client.proxy.generate_key(KeyGenerateBody(models=[], user_id="e2e-test-user"))
        resources.defer(lambda: client.proxy.delete_key(key))

        file: Final = unwrap(
            client.upload_file(
                content=render_jsonl(BLANK_S3_RAW_MODEL),
                form=FileUploadForm(purpose="batch", target_model_names="bedrock-blank-s3-batch"),
                key=key,
            )
        )
        resources.defer(lambda: cleanup_file(client, file.id, key=key))
        assert_file_object(file, provider="bedrock")

        created: Final = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        assert created.status_code < 400, (
            f"blank AWS_S3_ENCRYPTION_KEY_ID / AWS_S3_BUCKET_OWNER must be treated as "
            f"unset; Bedrock rejected the job: {created.body[:400]}"
        )
        require_successful_call(created)
        batch: Final = BatchObject.model_validate_json(created.body)
        resources.defer(lambda: cleanup_batch(client, batch.id, key=key))

        assert is_managed_id(batch.id), (
            f"blank-S3-env create via target_model_names must return a managed batch id, got {batch.id!r}"
        )
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"blank-S3-env batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)
