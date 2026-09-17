import copy
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.constants import MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES, MAX_S3_OBJECT_KEY_BYTES
from litellm.integrations.s3 import S3Logger, prompts_only_payload, resolve_s3_log_prompts_only

TEST_KMS_KEY_ARN = "arn:aws:kms:us-east-1:111122223333:key/test-key-id"
TEST_MESSAGES = [{"role": "user", "content": "Reply with exactly the word PINEAPPLE."}]
TEST_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "PINEAPPLE"}}]}


def _standard_logging_payload(response_id: str = "chatcmpl-test-id") -> dict:
    return {
        "id": response_id,
        "messages": copy.deepcopy(TEST_MESSAGES),
        "response": copy.deepcopy(TEST_RESPONSE),
        "metadata": {"user_api_key_team_alias": None},
    }


def _log_event_kwargs(response_id: str = "chatcmpl-test-id") -> dict:
    return {
        "litellm_params": {"metadata": {}},
        "standard_logging_object": _standard_logging_payload(response_id),
    }


def _run_log_event(
    callback_params: dict, response_id: str = "chatcmpl-test-id", log_kwargs: dict[str, object] | None = None
) -> MagicMock:
    original = litellm.s3_callback_params
    litellm.s3_callback_params = callback_params
    try:
        with patch("boto3.client") as mock_boto3_client:
            mock_s3_client = MagicMock()
            mock_boto3_client.return_value = mock_s3_client
            logger = S3Logger()
            logger.log_event(
                kwargs=_log_event_kwargs(response_id) if log_kwargs is None else log_kwargs,
                response_obj={"id": response_id},
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


def test_put_object_key_and_filename_are_bounded_for_an_oversized_response_id():
    """The sync logger bounds both the key and the Content-Disposition filename."""
    mock_s3_client = _run_log_event(
        {"s3_bucket_name": "test-bucket", "s3_region_name": "us-west-2", "s3_path": "logs"},
        response_id="resp_" + "A" * 1100,
    )

    put_object_kwargs = mock_s3_client.put_object.call_args.kwargs
    assert len(put_object_kwargs["Key"].encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert put_object_kwargs["Key"].startswith("logs/2026-07-30/time-12-00-00-000000_resp_")
    filename = put_object_kwargs["ContentDisposition"].removeprefix('inline; filename="').removesuffix('"')
    assert len(filename.encode("utf-8")) <= MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES


def test_put_object_keeps_the_configured_path_intact_when_only_the_id_has_to_shrink():
    """A long configured s3_path survives whole when the id can be shortened instead."""
    long_path = "litellm-prod-logs/" + "t" * 921
    mock_s3_client = _run_log_event(
        {"s3_bucket_name": "test-bucket", "s3_region_name": "us-west-2", "s3_path": long_path},
        response_id="resp_" + "B" * 100,
    )

    key = mock_s3_client.put_object.call_args.kwargs["Key"]
    assert key.startswith(long_path + "/2026-07-30/")
    assert len(key.encode("utf-8")) == MAX_S3_OBJECT_KEY_BYTES


def _uploaded_body(mock_s3_client: MagicMock) -> dict[str, object]:
    return json.loads(mock_s3_client.put_object.call_args.kwargs["Body"])


def test_log_event_prompts_only_drops_response_and_keeps_messages(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)
    log_kwargs = _log_event_kwargs()
    original_payload = copy.deepcopy(log_kwargs["standard_logging_object"])

    mock_s3_client = _run_log_event(
        {"s3_bucket_name": "test-bucket", "s3_region_name": "us-east-1", "s3_log_prompts_only": True},
        log_kwargs=log_kwargs,
    )

    body = _uploaded_body(mock_s3_client)
    assert body["messages"] == TEST_MESSAGES
    assert body["response"] is None
    assert body["id"] == "chatcmpl-test-id"
    assert log_kwargs["standard_logging_object"] == original_payload


def test_log_event_default_keeps_response(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)

    mock_s3_client = _run_log_event({"s3_bucket_name": "test-bucket", "s3_region_name": "us-east-1"})

    body = _uploaded_body(mock_s3_client)
    assert body["response"] == TEST_RESPONSE
    assert body["messages"] == TEST_MESSAGES


def test_log_event_reads_prompts_only_env_var_at_log_time(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)
    original = litellm.s3_callback_params
    litellm.s3_callback_params = {"s3_bucket_name": "test-bucket", "s3_region_name": "us-east-1"}
    try:
        with patch("boto3.client") as mock_boto3_client:
            mock_s3_client = MagicMock()
            mock_boto3_client.return_value = mock_s3_client
            logger = S3Logger()
            monkeypatch.setenv("S3_LOG_PROMPTS_ONLY", "true")
            logger.log_event(
                kwargs=_log_event_kwargs(),
                response_obj={"id": "chatcmpl-test-id"},
                start_time=datetime(2026, 7, 30, 12, 0, 0),
                end_time=datetime(2026, 7, 30, 12, 0, 1),
                print_verbose=lambda *args, **kwargs: None,
            )
    finally:
        litellm.s3_callback_params = original

    body = _uploaded_body(mock_s3_client)
    assert body["response"] is None
    assert body["messages"] == TEST_MESSAGES


def test_log_event_explicit_false_param_beats_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("S3_LOG_PROMPTS_ONLY", "true")

    mock_s3_client = _run_log_event(
        {"s3_bucket_name": "test-bucket", "s3_region_name": "us-east-1", "s3_log_prompts_only": False}
    )

    assert _uploaded_body(mock_s3_client)["response"] == TEST_RESPONSE


def test_s3_logger_init_does_not_mutate_global_callback_params(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_S3_BUCKET", "resolved-bucket")
    callback_params = {"s3_bucket_name": "os.environ/MY_S3_BUCKET", "s3_region_name": "us-east-1"}
    snapshot = copy.deepcopy(callback_params)
    original = litellm.s3_callback_params
    litellm.s3_callback_params = callback_params
    try:
        with patch("boto3.client"):
            logger = S3Logger()
    finally:
        litellm.s3_callback_params = original

    assert logger.bucket_name == "resolved-bucket"
    assert callback_params == snapshot


@pytest.mark.parametrize(
    "configured,env_value,expected",
    [
        (True, None, True),
        (False, "true", False),
        ("true", None, True),
        ("False", "true", False),
        ("1", None, True),
        ("0", None, False),
        (" yes ", None, True),
        (None, None, False),
        (None, "true", True),
        (None, "false", False),
        (None, "", False),
        ("", "true", False),
    ],
)
def test_resolve_s3_log_prompts_only(configured: object, env_value: str | None, expected: bool):
    environ = {} if env_value is None else {"S3_LOG_PROMPTS_ONLY": env_value}
    assert resolve_s3_log_prompts_only(configured, environ) is expected


def test_resolve_s3_log_prompts_only_unparseable_value_fails_toward_prompts_only():
    assert resolve_s3_log_prompts_only("enabled", {}) is True


def test_prompts_only_payload_returns_copy_with_response_cleared():
    payload = _standard_logging_payload()
    snapshot = copy.deepcopy(payload)

    stripped = prompts_only_payload(payload)

    assert stripped["response"] is None
    assert stripped["messages"] == TEST_MESSAGES
    assert stripped is not payload
    assert payload == snapshot
