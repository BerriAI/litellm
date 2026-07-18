"""Live e2e: Bedrock batch create under STS assume-role credentials.

Provisions a bedrock batch deployment whose litellm_params carry aws_role_name /
aws_session_name (the product path for role assumption) and runs the unified
file-upload + batch-create lifecycle. Success means the proxy assumed the role
and Bedrock accepted the job; a misconfigured role fails create with an AWS
auth error rather than silently falling back to the ambient key.
"""

from __future__ import annotations

import pytest

from batch_client import BatchClient, BatchCreateBody, BatchObject
from capabilities import batch_model_name
from e2e_config import require_env, unique_marker
from e2e_http import FileUploadForm, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from test_batches_e2e import (
    CREATED_BATCH_STATUSES,
    assert_batch_object,
    assert_file_object,
    quietly,
    render_jsonl,
)

pytestmark = pytest.mark.e2e

RAW_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _assume_role_params(role_arn: str, session_name: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=RAW_MODEL,
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
                content=render_jsonl(RAW_MODEL),
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
        assert batch.id.startswith("arn:aws:bedrock:") or batch.id, (
            f"bedrock batch id unexpected shape under assume-role: {batch.id!r}"
        )
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"assume-role batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)

        fetched = unwrap(client.retrieve_batch(batch.id, key=key))
        assert fetched.id == batch.id
