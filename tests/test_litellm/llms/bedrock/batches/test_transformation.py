"""
Regression tests for ``BedrockBatchesConfig`` (the BaseBatchesConfig
implementation for Bedrock model-invocation-job batches).

This file complements (does not duplicate):
  - ``test_batch_metadata_sanitization.py`` (covers
    ``_get_openai_compatible_batch_metadata`` exhaustively)
  - ``test_handler.py`` (covers the boto3-backed handler, not this transform)

Here we lock the pure transform logic in ``transformation.py``: request
construction (S3 input/output config, model id, job name, role ARN), the
AWS-JobStatus -> OpenAI-status mapping, timestamp parsing, retrieve-request
URL/ARN handling, and the error class. AWS auth/sigv4 is the only external seam
we mock; everything else runs for real.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
from litellm.types.utils import LiteLLMBatch, LlmProviders

# AWS JobStatus -> OpenAI BatchJobStatus, exactly as encoded in transformation.py
# (both transform_create_batch_response and transform_retrieve_batch_response).
STATUS_MAP = {
    "Submitted": "validating",
    "Validating": "validating",
    "Scheduled": "in_progress",
    "InProgress": "in_progress",
    "PartiallyCompleted": "completed",
    "Completed": "completed",
    "Failed": "failed",
    "Stopping": "cancelling",
    "Stopped": "cancelled",
    "Expired": "expired",
}

ARN = "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/abc1234567"


@pytest.fixture
def config():
    return BedrockBatchesConfig()


def _raw(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=body)


# --------------------------------------------------------------------------- #
# get_complete_batch_url
# --------------------------------------------------------------------------- #


def test_get_complete_batch_url_uses_region(config):
    url = config.get_complete_batch_url(
        api_base=None,
        api_key=None,
        model="anthropic.claude-3",
        optional_params={"aws_region_name": "eu-central-1"},
        litellm_params={},
        data={"input_file_id": "s3://b/k"},
    )
    assert url == "https://bedrock.eu-central-1.amazonaws.com/model-invocation-job"


# --------------------------------------------------------------------------- #
# transform_create_batch_request - request construction (sign_aws_request mocked)
# --------------------------------------------------------------------------- #


def test_create_request_builds_s3_input_output_and_arn(config):
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-deadbeef",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({"Authorization": "signed"}, b'{"x": 1}')
        result = config.transform_create_batch_request(
            model="anthropic.claude-3-5-sonnet",
            create_batch_data={
                "input_file_id": "s3://in-bucket/path/to/input.jsonl",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={
                "s3_output_bucket_name": "out-bucket",
                "aws_batch_role_arn": "arn:aws:iam::123:role/my-batch-role",
            },
        )

    bedrock_request = mock_sign.call_args.kwargs["data"]
    assert bedrock_request["modelId"] == "anthropic.claude-3-5-sonnet"
    assert bedrock_request["jobName"] == "litellm-batch-deadbeef"
    assert bedrock_request["roleArn"] == "arn:aws:iam::123:role/my-batch-role"
    assert (
        bedrock_request["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        == "s3://in-bucket/path/to/input.jsonl"
    )
    assert (
        bedrock_request["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        == "s3://out-bucket/litellm-batch-outputs/litellm-batch-deadbeef/"
    )
    # 24h completion window -> 24 hour timeout
    assert bedrock_request["timeoutDurationInHours"] == 24
    # signing was over the bedrock endpoint via POST
    assert mock_sign.call_args.kwargs["service_name"] == "bedrock"
    assert mock_sign.call_args.kwargs["method"] == "POST"
    assert mock_sign.call_args.kwargs["endpoint_url"] == (
        "https://bedrock.us-west-2.amazonaws.com/model-invocation-job"
    )
    # the transform returns the pre-signed envelope
    assert result["method"] == "POST"
    assert result["url"] == (
        "https://bedrock.us-west-2.amazonaws.com/model-invocation-job"
    )
    assert result["headers"] == {"Authorization": "signed"}


def test_create_request_defaults_output_bucket_to_input_bucket(config):
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-cafef00d",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://same-bucket/in.jsonl"},
            optional_params={},
            litellm_params={"aws_batch_role_arn": "arn:aws:iam::1:role/r"},
        )
    bedrock_request = mock_sign.call_args.kwargs["data"]
    assert (
        bedrock_request["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        == "s3://same-bucket/litellm-batch-outputs/litellm-batch-cafef00d/"
    )


def test_create_request_adds_kms_encryption_key_when_provided(config):
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-1",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={},
            litellm_params={
                "aws_batch_role_arn": "arn:aws:iam::1:role/r",
                "s3_encryption_key_id": "kms-key-123",
            },
        )
    s3out = mock_sign.call_args.kwargs["data"]["outputDataConfig"][
        "s3OutputDataConfig"
    ]
    assert s3out["s3EncryptionKeyId"] == "kms-key-123"


def test_create_request_omits_kms_key_when_absent(config):
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-1",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign, patch(
        "litellm.llms.bedrock.batches.transformation.get_secret_str",
        return_value=None,
    ):
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={},
            litellm_params={"aws_batch_role_arn": "arn:aws:iam::1:role/r"},
        )
    s3out = mock_sign.call_args.kwargs["data"]["outputDataConfig"][
        "s3OutputDataConfig"
    ]
    assert "s3EncryptionKeyId" not in s3out


def test_create_request_missing_input_file_id_raises(config):
    with pytest.raises(ValueError, match="input_file_id is required"):
        config.transform_create_batch_request(
            model="m",
            create_batch_data={},
            optional_params={},
            litellm_params={"aws_batch_role_arn": "arn:aws:iam::1:role/r"},
        )


def test_create_request_missing_role_arn_raises(config, monkeypatch):
    monkeypatch.delenv("AWS_BATCH_ROLE_ARN", raising=False)
    with pytest.raises(ValueError, match="IAM role ARN is required"):
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={},
            litellm_params={},
        )


def test_create_request_role_arn_from_env(config, monkeypatch):
    monkeypatch.setenv("AWS_BATCH_ROLE_ARN", "arn:aws:iam::9:role/env-role")
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-1",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={},
            litellm_params={},
        )
    assert (
        mock_sign.call_args.kwargs["data"]["roleArn"]
        == "arn:aws:iam::9:role/env-role"
    )


def test_create_request_missing_model_raises(config):
    with pytest.raises(ValueError, match="Could not determine Bedrock model ID"):
        config.transform_create_batch_request(
            model="",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={},
            litellm_params={"aws_batch_role_arn": "arn:aws:iam::1:role/r"},
        )


def test_create_request_signs_with_credentials_from_litellm_params(config):
    """Regression: the proxy's model-based routing forwards deployment AWS
    credentials via ``litellm_params`` while ``base_llm_http_handler.create_batch``
    calls this transform with ``optional_params={}``. The signer reads credentials
    from ``optional_params``, so unless the transform merges ``litellm_params`` in,
    boto3 sees no keys and raises "Unable to locate credentials"; region resolution
    also has to see ``aws_region_name`` from ``litellm_params``."""
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-1",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={},
            litellm_params={
                "aws_batch_role_arn": "arn:aws:iam::1:role/r",
                "aws_access_key_id": "AKIA-DEPLOYMENT",
                "aws_secret_access_key": "secret-deployment",
                "aws_region_name": "ap-south-1",
            },
        )
    signing_params = mock_sign.call_args.kwargs["optional_params"]
    assert signing_params["aws_access_key_id"] == "AKIA-DEPLOYMENT"
    assert signing_params["aws_secret_access_key"] == "secret-deployment"
    assert mock_sign.call_args.kwargs["endpoint_url"] == (
        "https://bedrock.ap-south-1.amazonaws.com/model-invocation-job"
    )


def test_create_request_optional_params_win_over_litellm_params_for_signing(config):
    """Per-request ``optional_params`` must override deployment ``litellm_params``
    when both carry the same credential field."""
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-1",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={"input_file_id": "s3://b/in.jsonl"},
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={
                "aws_batch_role_arn": "arn:aws:iam::1:role/r",
                "aws_region_name": "ap-south-1",
            },
        )
    assert mock_sign.call_args.kwargs["endpoint_url"] == (
        "https://bedrock.us-west-2.amazonaws.com/model-invocation-job"
    )


def test_create_request_no_timeout_for_non_24h_window(config):
    with patch.object(
        config.common_utils,
        "generate_unique_job_name",
        return_value="litellm-batch-1",
    ), patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({}, b"{}")
        config.transform_create_batch_request(
            model="m",
            create_batch_data={
                "input_file_id": "s3://b/in.jsonl",
                "completion_window": "48h",
            },
            optional_params={},
            litellm_params={"aws_batch_role_arn": "arn:aws:iam::1:role/r"},
        )
    assert "timeoutDurationInHours" not in mock_sign.call_args.kwargs["data"]


# --------------------------------------------------------------------------- #
# transform_create_batch_response - status mapping + LiteLLMBatch shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bedrock_status,openai_status", list(STATUS_MAP.items()))
def test_create_response_status_mapping(config, bedrock_status, openai_status):
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": bedrock_status}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.status == openai_status
    assert out.id == ARN
    assert out.object == "batch"


def test_create_response_unknown_status_falls_back_to_validating(config):
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "SomeFutureStatus"}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.status == "validating"


def test_create_response_default_status_when_missing(config):
    # status defaults to "Submitted" -> "validating"
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.status == "validating"


def test_create_response_in_progress_sets_in_progress_at(config):
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "InProgress"}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.status == "in_progress"
    assert isinstance(out.in_progress_at, int)


def test_create_response_non_in_progress_leaves_in_progress_at_none(config):
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "Submitted"}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.in_progress_at is None


def test_create_response_uses_original_request_fields(config):
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "Submitted"}),
        logging_obj=MagicMock(),
        litellm_params={
            "original_batch_request": {
                "endpoint": "/v1/embeddings",
                "input_file_id": "s3://b/in.jsonl",
                "completion_window": "24h",
                "metadata": {"user": "alice"},
            }
        },
    )
    assert out.endpoint == "/v1/embeddings"
    assert out.input_file_id == "s3://b/in.jsonl"
    assert out.metadata == {"user": "alice"}


def test_create_response_default_endpoint_when_no_original_request(config):
    out = config.transform_create_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "Submitted"}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.endpoint == "/v1/chat/completions"
    assert out.completion_window == "24h"


def test_create_response_raises_on_unparseable_body(config):
    bad = httpx.Response(status_code=200, text="not-json")
    with pytest.raises(ValueError, match="Failed to parse Bedrock batch response"):
        config.transform_create_batch_response(
            model=None,
            raw_response=bad,
            logging_obj=MagicMock(),
            litellm_params={},
        )


# --------------------------------------------------------------------------- #
# transform_retrieve_batch_request - ARN validation + URL construction
# --------------------------------------------------------------------------- #


def test_retrieve_request_builds_encoded_arn_url(config):
    with patch.object(config.common_utils, "sign_aws_request") as mock_sign:
        mock_sign.return_value = ({"Authorization": "signed"}, b"")
        result = config.transform_retrieve_batch_request(
            batch_id=ARN, optional_params={}, litellm_params={}
        )
    # ARN is URL-encoded (colons and slashes escaped) into the path
    assert result["method"] == "GET"
    assert result["data"] is None
    assert result["headers"] == {"Authorization": "signed"}
    assert result["url"].startswith(
        "https://bedrock.us-west-2.amazonaws.com/model-invocation-job/"
    )
    assert "%3A" in result["url"]  # colon encoded
    assert "%2F" in result["url"]  # slash encoded
    assert mock_sign.call_args.kwargs["method"] == "GET"
    assert mock_sign.call_args.kwargs["data"] == {}


def test_retrieve_request_rejects_non_arn(config):
    with pytest.raises(ValueError, match="Expected ARN"):
        config.transform_retrieve_batch_request(
            batch_id="abc1234567", optional_params={}, litellm_params={}
        )


def test_retrieve_request_rejects_short_arn(config):
    with pytest.raises(ValueError, match="Invalid ARN format"):
        config.transform_retrieve_batch_request(
            batch_id="arn:aws:bedrock:us-west-2", optional_params={}, litellm_params={}
        )


def test_retrieve_request_rejects_bad_region(config):
    bad = "arn:aws:bedrock:US_WEST:123:model-invocation-job/x"
    with pytest.raises(ValueError, match="Invalid region in ARN"):
        config.transform_retrieve_batch_request(
            batch_id=bad, optional_params={}, litellm_params={}
        )


# --------------------------------------------------------------------------- #
# transform_retrieve_batch_response - status, timestamps, files, errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bedrock_status,openai_status", list(STATUS_MAP.items()))
def test_retrieve_response_status_mapping(config, bedrock_status, openai_status):
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": bedrock_status}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.status == openai_status


def test_retrieve_response_unknown_status_falls_back_to_validating(config):
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "NewStatus"}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.status == "validating"


def test_retrieve_response_extracts_file_configs(config):
    body = {
        "jobArn": ARN,
        "status": "Completed",
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://b/in.jsonl"}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out/"}},
    }
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.input_file_id == "s3://b/in.jsonl"
    assert out.output_file_id == "s3://b/out/"


def test_retrieve_response_parses_timestamps_for_completed(config):
    body = {
        "jobArn": ARN,
        "status": "Completed",
        "submitTime": "2026-04-28T12:00:00Z",
        "endTime": "2026-04-28T12:30:00Z",
        "jobExpirationTime": "2026-05-28T12:00:00Z",
    }
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    import datetime

    expect_created = int(
        datetime.datetime.fromisoformat("2026-04-28T12:00:00+00:00").timestamp()
    )
    expect_completed = int(
        datetime.datetime.fromisoformat("2026-04-28T12:30:00+00:00").timestamp()
    )
    expect_expires = int(
        datetime.datetime.fromisoformat("2026-05-28T12:00:00+00:00").timestamp()
    )
    assert out.created_at == expect_created
    assert out.completed_at == expect_completed
    assert out.expires_at == expect_expires
    # completed -> not failed/cancelled, no in_progress timestamp
    assert out.failed_at is None
    assert out.cancelled_at is None
    assert out.in_progress_at is None


def test_retrieve_response_failed_sets_failed_at_from_end_time(config):
    body = {
        "jobArn": ARN,
        "status": "Failed",
        "endTime": "2026-04-28T12:30:00Z",
    }
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    import datetime

    assert out.failed_at == int(
        datetime.datetime.fromisoformat("2026-04-28T12:30:00+00:00").timestamp()
    )
    assert out.completed_at is None
    assert out.cancelled_at is None


def test_retrieve_response_stopped_sets_cancelled_at(config):
    body = {"jobArn": ARN, "status": "Stopped", "endTime": "2026-04-28T12:30:00Z"}
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    import datetime

    assert out.cancelled_at == int(
        datetime.datetime.fromisoformat("2026-04-28T12:30:00+00:00").timestamp()
    )
    assert out.completed_at is None
    assert out.failed_at is None


def test_retrieve_response_in_progress_sets_in_progress_at_from_last_modified(config):
    body = {
        "jobArn": ARN,
        "status": "InProgress",
        "lastModifiedTime": "2026-04-28T12:15:00Z",
    }
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    import datetime

    assert out.in_progress_at == int(
        datetime.datetime.fromisoformat("2026-04-28T12:15:00+00:00").timestamp()
    )


def test_retrieve_response_invalid_timestamp_becomes_none(config):
    body = {
        "jobArn": ARN,
        "status": "Completed",
        "submitTime": "not-a-timestamp",
        "endTime": "also-bad",
    }
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    # created_at falls back to int(time.time()) when submitTime unparseable
    assert isinstance(out.created_at, int)
    assert out.completed_at is None


def test_retrieve_response_builds_errors_from_message(config):
    body = {"jobArn": ARN, "status": "Failed", "message": "validation failed"}
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body, status_code=400),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.errors is not None
    assert out.errors.data[0].message == "validation failed"
    assert out.errors.data[0].code == "400"


def test_retrieve_response_no_errors_when_no_message(config):
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw({"jobArn": ARN, "status": "Completed"}),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.errors is None


def test_retrieve_response_enriches_metadata(config):
    body = {
        "jobArn": ARN,
        "status": "Completed",
        "jobName": "litellm-batch-1",
        "modelId": "anthropic.claude-3",
        "roleArn": "arn:aws:iam::1:role/r",
        "timeoutDurationInHours": 24,
        "vpcConfig": {"subnetIds": ["subnet-1"]},
        "clientRequestToken": None,
    }
    out = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_raw(body),
        logging_obj=MagicMock(),
        litellm_params={},
    )
    assert out.metadata["jobName"] == "litellm-batch-1"
    assert out.metadata["modelId"] == "anthropic.claude-3"
    assert out.metadata["roleArn"] == "arn:aws:iam::1:role/r"
    # non-string scalar is stringified
    assert out.metadata["timeoutDurationInHours"] == "24"
    # dict/list serialized to JSON string
    assert out.metadata["vpcConfig"] == '{"subnetIds": ["subnet-1"]}'
    # None-valued fields dropped
    assert "clientRequestToken" not in out.metadata


def test_retrieve_response_raises_on_unparseable_body(config):
    bad = httpx.Response(status_code=200, text="<<<not json>>>")
    with pytest.raises(ValueError, match="Failed to parse Bedrock batch response"):
        config.transform_retrieve_batch_response(
            model=None,
            raw_response=bad,
            logging_obj=MagicMock(),
            litellm_params={},
        )


# --------------------------------------------------------------------------- #
# get_error_class + custom_llm_provider
# --------------------------------------------------------------------------- #


def test_get_error_class_returns_bedrock_error(config):
    err = config.get_error_class(
        error_message="throttled", status_code=429, headers={}
    )
    assert isinstance(err, Exception)
    assert err.status_code == 429
    assert "throttled" in str(err)


def test_custom_llm_provider_is_bedrock(config):
    assert config.custom_llm_provider == LlmProviders.BEDROCK


def test_validate_environment_passes_headers_through(config):
    headers = {"X-Custom": "v"}
    out = config.validate_environment(
        headers=headers,
        model="m",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert out == headers


# --------------------------------------------------------------------------- #
# Shared BaseBatchesConfig contract suite.
# --------------------------------------------------------------------------- #

from tests.test_litellm.llms.base_llm.batches.base_batches_config_test import (  # noqa: E402
    BatchesConfigContractTests,
)


class TestBedrockBatchesContract(BatchesConfigContractTests):
    def make_config(self):
        return BedrockBatchesConfig()

    expected_provider = LlmProviders.BEDROCK
    # Bedrock builds a real create request (no NotImplementedError); the
    # bedrock-specific create tests above cover the request/response shape.
    supports_create = True
    # Retrieve responses are parsed in this transformation layer
    # (transform_retrieve_batch_response).
    supports_retrieve_response = True

    def sample_retrieve_response_body(self) -> dict:
        return {
            "jobArn": ARN,
            "status": "Completed",
            "submitTime": "2026-04-28T12:00:00Z",
            "endTime": "2026-04-28T12:30:00Z",
            "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://b/in.jsonl"}},
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out/"}},
        }

    expected_retrieve_batch_id = ARN
    expected_retrieve_status = "completed"
