from datetime import datetime
from unittest.mock import MagicMock, patch

import litellm
from litellm.integrations.s3 import S3Logger

TEST_KMS_KEY_ARN = "arn:aws:kms:us-east-1:111122223333:key/test-key-id"


def _standard_logging_payload() -> dict:
    return {
        "id": "chatcmpl-test-id",
        "metadata": {"user_api_key_team_alias": None},
    }


def _log_event_kwargs() -> dict:
    return {
        "litellm_params": {"metadata": {}},
        "standard_logging_object": _standard_logging_payload(),
    }


def _run_log_event(callback_params: dict) -> MagicMock:
    original = litellm.s3_callback_params
    litellm.s3_callback_params = callback_params
    try:
        with patch("boto3.client") as mock_boto3_client:
            mock_s3_client = MagicMock()
            mock_boto3_client.return_value = mock_s3_client
            logger = S3Logger()
            logger.log_event(
                kwargs=_log_event_kwargs(),
                response_obj={},
                start_time=datetime(2026, 7, 30, 12, 0, 0),
                end_time=datetime(2026, 7, 30, 12, 0, 1),
                print_verbose=lambda *args, **kwargs: None,
            )
        return mock_s3_client
    finally:
        litellm.s3_callback_params = original


def test_put_object_includes_sse_kms_params_when_configured():
    """
    When s3_server_side_encryption and s3_sse_kms_key_id are set in
    s3_callback_params, put_object must receive ServerSideEncryption and
    SSEKMSKeyId so objects land encrypted with the customer-managed key.
    """
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_server_side_encryption": "aws:kms",
            "s3_sse_kms_key_id": TEST_KMS_KEY_ARN,
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert put_object_kwargs["ServerSideEncryption"] == "aws:kms"
    assert put_object_kwargs["SSEKMSKeyId"] == TEST_KMS_KEY_ARN


def test_put_object_supports_sse_s3_without_key_id():
    """SSE-S3 (AES256) needs only ServerSideEncryption, no key id."""
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_server_side_encryption": "AES256",
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert put_object_kwargs["ServerSideEncryption"] == "AES256"
    assert "SSEKMSKeyId" not in put_object_kwargs


def test_put_object_omits_sse_params_by_default():
    """Without SSE config, put_object kwargs must stay unchanged."""
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert "ServerSideEncryption" not in put_object_kwargs
    assert "SSEKMSKeyId" not in put_object_kwargs


def test_put_object_infers_aws_kms_when_only_key_id_set():
    """A key id without an algorithm must infer aws:kms instead of sending an invalid request."""
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_sse_kms_key_id": TEST_KMS_KEY_ARN,
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert put_object_kwargs["ServerSideEncryption"] == "aws:kms"
    assert put_object_kwargs["SSEKMSKeyId"] == TEST_KMS_KEY_ARN


def test_put_object_drops_key_id_when_algorithm_is_not_kms():
    """AES256 plus a key id is invalid for S3; the key id must be dropped, not sent."""
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_server_side_encryption": "AES256",
            "s3_sse_kms_key_id": TEST_KMS_KEY_ARN,
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert put_object_kwargs["ServerSideEncryption"] == "AES256"
    assert "SSEKMSKeyId" not in put_object_kwargs


def test_non_string_algorithm_is_dropped_and_valid_key_id_is_rescued():
    """
    A YAML boolean in s3_server_side_encryption must not crash logger init and
    must not discard the valid key id; aws:kms is inferred from the key id.
    """
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_server_side_encryption": True,
            "s3_sse_kms_key_id": TEST_KMS_KEY_ARN,
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert put_object_kwargs["ServerSideEncryption"] == "aws:kms"
    assert put_object_kwargs["SSEKMSKeyId"] == TEST_KMS_KEY_ARN


def test_non_string_key_id_is_dropped_and_valid_algorithm_is_kept():
    """A mistyped key id (unquoted YAML number) must not disable the valid algorithm."""
    mock_s3_client = _run_log_event(
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_server_side_encryption": "aws:kms",
            "s3_sse_kms_key_id": 12345,
        }
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert put_object_kwargs["ServerSideEncryption"] == "aws:kms"
    assert "SSEKMSKeyId" not in put_object_kwargs
