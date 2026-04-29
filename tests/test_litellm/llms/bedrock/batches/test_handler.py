"""Unit tests for ``BedrockBatchesHandler._handle_model_invocation_job_status``.

These cover the upstream support for retrieving Bedrock bulk batch jobs
(``arn:aws:bedrock:<region>:<acct>:model-invocation-job/<id>``) — the ARN
type returned by ``CreateModelInvocationJob``. We mock the boto3 client so
the tests don't hit AWS.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.bedrock.batches.handler import (  # noqa: E402
    BedrockBatchesHandler,
    _extract_job_id_from_arn,
    _extract_region_from_bedrock_arn,
    _predict_output_file_uri,
)

JOB_ID = "abc1234567"
JOB_ARN = f"arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/{JOB_ID}"
INPUT_URI = "s3://my-bucket/inputs/qwen3-235b-a22b-2507-batch.jsonl"
OUTPUT_PREFIX = "s3://my-bucket/litellm-batch-outputs/litellm-bedrock-files-qwen-uuid/"
SUBMIT_TIME = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2026, 4, 28, 12, 30, 0, tzinfo=timezone.utc)


def _fake_boto3_response(status: str = "Completed", end_time=END_TIME):
    return {
        "jobArn": JOB_ARN,
        "jobName": "litellm-bedrock-files-qwen-uuid",
        "modelId": "bedrock/qwen.qwen3-235b-a22b-2507-v1:0",
        "status": status,
        "submitTime": SUBMIT_TIME,
        "lastModifiedTime": end_time,
        "endTime": end_time,
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": INPUT_URI}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": OUTPUT_PREFIX}},
    }


@pytest.fixture
def patched_boto3():
    """Yield a stub bedrock client whose `get_model_invocation_job` is a MagicMock."""
    fake_client = MagicMock()
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response()
    with (
        patch("boto3.client", return_value=fake_client) as boto_client_factory,
        patch(
            "litellm.llms.bedrock.batches.transformation.BedrockBatchesConfig.get_credentials",
            return_value=MagicMock(access_key="AKIA", secret_key="SECRET", token=None),
        ),
    ):
        yield fake_client, boto_client_factory


def test_extract_region_from_arn():
    assert _extract_region_from_bedrock_arn(JOB_ARN) == "us-west-2"
    assert _extract_region_from_bedrock_arn("arn:aws:bedrock::123:foo/bar") is None
    assert _extract_region_from_bedrock_arn("not-an-arn") is None


def test_extract_job_id_from_arn():
    assert _extract_job_id_from_arn(JOB_ARN) == JOB_ID
    assert (
        _extract_job_id_from_arn("arn:aws:bedrock:us-west-2:1:async-invoke/x") is None
    )


def test_predict_output_file_uri_happy_path():
    expected = f"{OUTPUT_PREFIX}{JOB_ID}/qwen3-235b-a22b-2507-batch.jsonl.out"
    assert _predict_output_file_uri(OUTPUT_PREFIX, INPUT_URI, JOB_ID) == expected


def test_predict_output_file_uri_adds_trailing_slash():
    prefix_no_slash = OUTPUT_PREFIX.rstrip("/")
    expected = f"{OUTPUT_PREFIX}{JOB_ID}/qwen3-235b-a22b-2507-batch.jsonl.out"
    assert _predict_output_file_uri(prefix_no_slash, INPUT_URI, JOB_ID) == expected


@pytest.mark.parametrize(
    "missing_arg",
    [
        ("", INPUT_URI, JOB_ID),
        (OUTPUT_PREFIX, "", JOB_ID),
        (OUTPUT_PREFIX, INPUT_URI, None),
    ],
)
def test_predict_output_file_uri_returns_none_when_missing_input(missing_arg):
    assert _predict_output_file_uri(*missing_arg) is None


def test_handle_model_invocation_job_status_completed(patched_boto3):
    fake_client, boto_client_factory = patched_boto3

    batch = BedrockBatchesHandler._handle_model_invocation_job_status(batch_id=JOB_ARN)

    fake_client.get_model_invocation_job.assert_called_once_with(jobIdentifier=JOB_ARN)

    # Region should be sniffed from the ARN.
    _, kwargs = boto_client_factory.call_args
    assert kwargs["region_name"] == "us-west-2"

    assert batch.id == JOB_ARN
    assert batch.status == "completed"
    assert batch.input_file_id == INPUT_URI
    expected_out = f"{OUTPUT_PREFIX}{JOB_ID}/qwen3-235b-a22b-2507-batch.jsonl.out"
    assert batch.output_file_id == expected_out
    assert batch.completed_at == int(END_TIME.timestamp())
    assert batch.failed_at is None
    assert batch.cancelled_at is None
    # Per-record counts aren't reported by GetModelInvocationJob, so we leave
    # them zeroed; consumers should parse manifest.json.out for accurate counts.
    assert batch.request_counts.total == 0
    assert batch.metadata["job_arn"] == JOB_ARN
    assert batch.metadata["output_file_uri"] == expected_out
    assert batch.metadata["output_s3_uri"] == OUTPUT_PREFIX


@pytest.mark.parametrize(
    "bedrock_status,openai_status",
    [
        ("Submitted", "validating"),
        ("Validating", "validating"),
        ("Scheduled", "validating"),
        ("InProgress", "in_progress"),
        ("Stopping", "cancelling"),
        ("Stopped", "cancelled"),
        ("Completed", "completed"),
        ("PartiallyCompleted", "completed"),
        ("Failed", "failed"),
        ("Expired", "expired"),
        # Unknown/unmapped Bedrock status falls back to "in_progress" so we
        # don't 500 on a future AWS-side enum addition.
        ("MyBrandNewStatus", "in_progress"),
    ],
)
def test_status_mapping(patched_boto3, bedrock_status, openai_status):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(
        status=bedrock_status
    )

    batch = BedrockBatchesHandler._handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == openai_status
    # output_file_id is only populated for terminal-completed jobs, so callers
    # don't accidentally try to download a non-existent file mid-run.
    if openai_status == "completed":
        assert batch.output_file_id is not None
    else:
        assert batch.output_file_id is None


def test_explicit_region_overrides_arn(patched_boto3):
    _, boto_client_factory = patched_boto3
    BedrockBatchesHandler._handle_model_invocation_job_status(
        batch_id=JOB_ARN, aws_region_name="eu-central-1"
    )
    _, kwargs = boto_client_factory.call_args
    assert kwargs["region_name"] == "eu-central-1"


def test_failure_message_propagates(patched_boto3):
    fake_client, _ = patched_boto3
    failed_response = _fake_boto3_response(status="Failed")
    failed_response["message"] = "Input file failed validation"
    fake_client.get_model_invocation_job.return_value = failed_response

    batch = BedrockBatchesHandler._handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == "failed"
    assert batch.failed_at == int(END_TIME.timestamp())
    assert batch.metadata["failure_message"] == "Input file failed validation"
