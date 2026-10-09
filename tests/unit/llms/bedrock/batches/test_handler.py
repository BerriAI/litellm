"""Unit tests for ``BedrockBatchesHandler.handle_model_invocation_job_status``.

These cover the upstream support for retrieving Bedrock bulk batch jobs
(``arn:aws:bedrock:<region>:<acct>:model-invocation-job/<id>``) — the ARN
type returned by ``CreateModelInvocationJob``. We mock the boto3 client so
the tests don't hit AWS.
"""

from __future__ import annotations

import json
import json as json_module
import os
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse


import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.bedrock.batches.handler import (  # noqa: E402
    BedrockBatchesHandler,
    _extract_job_id_from_arn,
    _extract_region_from_bedrock_arn,
    _predict_output_file_uri,
    _to_epoch,
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


def test_extract_region_swallows_unexpected_split_errors():
    """Defensive `except Exception` branch — anything that isn't a plain str
    should fall through to ``None`` rather than blow up."""

    class WeirdArn:
        def split(self, _sep):
            raise RuntimeError("boom")

    assert _extract_region_from_bedrock_arn(WeirdArn()) is None  # type: ignore[arg-type]


def test_predict_output_file_uri_returns_none_for_directory_input_uri():
    """Input URI ending in `/` has an empty basename — we must bail rather
    than emit ``<prefix>/<job-id>/.out``."""
    assert (
        _predict_output_file_uri(OUTPUT_PREFIX, "s3://bucket/inputs/", JOB_ID) is None
    )


_DT = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (1730000000, 1730000000),
        (1730000000.5, 1730000000),
        (_DT, int(_DT.timestamp())),
        ("2026-04-28T12:00:00Z", None),  # strings aren't supported -> None
    ],
)
def test_to_epoch_handles_supported_types(value, expected):
    assert _to_epoch(value) == expected


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

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

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
    assert batch.request_counts is None
    assert batch.metadata["job_arn"] == JOB_ARN
    assert batch.metadata["output_file_uri"] == expected_out
    assert batch.metadata["output_s3_uri"] == OUTPUT_PREFIX


@pytest.mark.parametrize("success_count,error_count", [(100, 0), (86, 14)])
def test_completed_job_maps_provider_record_counts(patched_boto3, success_count, error_count):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = {
        **_fake_boto3_response(),
        "totalRecordCount": 100,
        "successRecordCount": success_count,
        "errorRecordCount": error_count,
    }

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.request_counts is not None
    assert (batch.request_counts.total, batch.request_counts.completed, batch.request_counts.failed) == (
        100,
        success_count,
        error_count,
    )


def test_missing_record_counts_leave_request_counts_none(patched_boto3):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response()

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.request_counts is None


def test_total_without_success_count_leaves_request_counts_none(patched_boto3):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = {**_fake_boto3_response(), "totalRecordCount": 100}

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.request_counts is None


def test_missing_error_count_maps_to_zero_failed(patched_boto3):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = {
        **_fake_boto3_response(),
        "totalRecordCount": 100,
        "successRecordCount": 100,
    }

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.request_counts is not None
    assert (batch.request_counts.total, batch.request_counts.completed, batch.request_counts.failed) == (100, 100, 0)


@pytest.mark.parametrize(
    "bedrock_status,openai_status",
    [
        ("Submitted", "validating"),
        ("Validating", "validating"),
        ("Scheduled", "in_progress"),
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
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status=bedrock_status)

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == openai_status
    # output_file_id is only populated for terminal-completed jobs, so callers
    # don't accidentally try to download a non-existent file mid-run.
    if openai_status == "completed":
        assert batch.output_file_id is not None
    else:
        assert batch.output_file_id is None


def test_explicit_region_overrides_arn(patched_boto3):
    _, boto_client_factory = patched_boto3
    BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN, aws_region_name="eu-central-1")
    _, kwargs = boto_client_factory.call_args
    assert kwargs["region_name"] == "eu-central-1"


def test_failure_message_propagates(patched_boto3):
    fake_client, _ = patched_boto3
    failed_response = _fake_boto3_response(status="Failed")
    failed_response["message"] = "Input file failed validation"
    fake_client.get_model_invocation_job.return_value = failed_response

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == "failed"
    assert batch.failed_at == int(END_TIME.timestamp())
    assert batch.metadata["failure_message"] == "Input file failed validation"


def test_completed_with_unpredictable_output_uri_stays_none(patched_boto3):
    """
    Regression guard for the original NoSuchKey bug: if Bedrock's response is
    missing pieces we need to compute the per-job output file path (here, the
    input s3Uri), `output_file_id` must stay `None` rather than fall back to
    the bare prefix. Falling back to the prefix is what produced the original
    NoSuchKey error this PR fixes.
    """
    fake_client, _ = patched_boto3
    incomplete_response = _fake_boto3_response(status="Completed")
    incomplete_response["inputDataConfig"] = {"s3InputDataConfig": {"s3Uri": ""}}
    fake_client.get_model_invocation_job.return_value = incomplete_response

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == "completed"
    # output_file_id MUST be None (not the bare prefix) — that's the whole
    # point of this regression test. Callers branch on this field.
    assert batch.output_file_id is None
    # The metadata field uses "" because OpenAI Batch metadata is dict[str, str];
    # callers should branch on `output_file_id` (above) instead.
    assert batch.metadata["output_file_uri"] == ""
    # The bare prefix is still preserved in metadata so callers can list it.
    assert batch.metadata["output_s3_uri"] == OUTPUT_PREFIX


def test_cancelled_status_sets_cancelled_at(patched_boto3):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="Stopped")

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == "cancelled"
    assert batch.cancelled_at == int(END_TIME.timestamp())
    assert batch.completed_at is None
    assert batch.failed_at is None
    assert batch.expired_at is None


def test_expired_status_sets_expired_at(patched_boto3):
    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="Expired")

    batch = BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)

    assert batch.status == "expired"
    assert batch.expired_at == int(END_TIME.timestamp())
    assert batch.completed_at is None
    assert batch.failed_at is None
    assert batch.cancelled_at is None


def test_logging_obj_pre_and_post_call_invoked(patched_boto3):
    """`pre_call` / `post_call` get called with sensible payloads when a
    `logging_obj` is supplied."""
    _, _ = patched_boto3
    logging_obj = MagicMock()

    BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN, logging_obj=logging_obj)

    logging_obj.pre_call.assert_called_once()
    logging_obj.post_call.assert_called_once()

    pre_kwargs = logging_obj.pre_call.call_args.kwargs
    assert pre_kwargs["input"] == JOB_ARN
    assert pre_kwargs["additional_args"]["complete_input_dict"] == {"jobIdentifier": JOB_ARN}
    # Logged URL must use the bare job id, not the full ARN, so it doesn't
    # double the `model-invocation-job/` segment or embed colons in the path.
    assert pre_kwargs["additional_args"]["api_base"] == (
        f"https://bedrock.us-west-2.amazonaws.com/model-invocation-job/{JOB_ID}"
    )

    post_kwargs = logging_obj.post_call.call_args.kwargs
    assert post_kwargs["input"] == JOB_ARN
    assert post_kwargs["original_response"]["jobArn"] == JOB_ARN


def test_missing_boto3_raises_helpful_import_error():
    """If boto3 isn't installed we should raise a clear, actionable
    ImportError rather than letting a NameError escape."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="pip install boto3"):
            BedrockBatchesHandler.handle_model_invocation_job_status(batch_id=JOB_ARN)


def test_logging_url_uses_bare_id_when_only_id_passed(patched_boto3):
    """If the caller passes just the trailing job id (also valid for
    `GetModelInvocationJob`), the logged URL should use it as-is."""
    _, _ = patched_boto3
    logging_obj = MagicMock()

    BedrockBatchesHandler.handle_model_invocation_job_status(
        batch_id=JOB_ID, aws_region_name="us-west-2", logging_obj=logging_obj
    )

    pre_kwargs = logging_obj.pre_call.call_args.kwargs
    assert pre_kwargs["additional_args"]["api_base"] == (
        f"https://bedrock.us-west-2.amazonaws.com/model-invocation-job/{JOB_ID}"
    )


def test_cancel_batch_stops_job_and_returns_mapped_status(patched_boto3):
    fake_client, boto_client_factory = patched_boto3
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="Stopping")

    batch = BedrockBatchesHandler.cancel_batch(batch_id=JOB_ARN)

    fake_client.stop_model_invocation_job.assert_called_once_with(jobIdentifier=JOB_ARN)
    _, kwargs = boto_client_factory.call_args
    assert kwargs["region_name"] == "us-west-2"
    assert batch.status == "cancelling"


def test_cancel_batch_tolerates_already_terminal_job(patched_boto3):
    from botocore.exceptions import ClientError

    fake_client, _ = patched_boto3
    fake_client.stop_model_invocation_job.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Job is already in a terminal state"}},
        "StopModelInvocationJob",
    )
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="Stopped")

    batch = BedrockBatchesHandler.cancel_batch(batch_id=JOB_ARN)

    assert batch.status == "cancelled"


def test_cancel_batch_tolerates_conflict_on_already_stopped_job(patched_boto3):
    from botocore.exceptions import ClientError

    fake_client, _ = patched_boto3
    fake_client.stop_model_invocation_job.side_effect = ClientError(
        {"Error": {"Code": "ConflictException", "Message": "Job cannot be stopped in its current state"}},
        "StopModelInvocationJob",
    )
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="Stopped")

    batch = BedrockBatchesHandler.cancel_batch(batch_id=JOB_ARN)

    assert batch.status == "cancelled"


def test_cancel_batch_reraises_conflict_when_job_not_terminal(patched_boto3):
    from botocore.exceptions import ClientError

    fake_client, _ = patched_boto3
    fake_client.stop_model_invocation_job.side_effect = ClientError(
        {"Error": {"Code": "ConflictException", "Message": "Operation conflicts with current job state"}},
        "StopModelInvocationJob",
    )
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="InProgress")

    with pytest.raises(ClientError):
        BedrockBatchesHandler.cancel_batch(batch_id=JOB_ARN)


def test_cancel_batch_reraises_validation_error_when_job_not_terminal(patched_boto3):
    from botocore.exceptions import ClientError

    fake_client, _ = patched_boto3
    fake_client.stop_model_invocation_job.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Cannot stop job in current state"}},
        "StopModelInvocationJob",
    )
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="InProgress")

    with pytest.raises(ClientError):
        BedrockBatchesHandler.cancel_batch(batch_id=JOB_ARN)


def test_cancel_batch_reraises_other_client_errors(patched_boto3):
    from botocore.exceptions import ClientError

    fake_client, _ = patched_boto3
    fake_client.stop_model_invocation_job.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
        "StopModelInvocationJob",
    )

    with pytest.raises(ClientError):
        BedrockBatchesHandler.cancel_batch(batch_id=JOB_ARN)

    fake_client.get_model_invocation_job.assert_not_called()


def test_litellm_cancel_batch_dispatches_to_bedrock(patched_boto3):
    import litellm

    fake_client, _ = patched_boto3
    fake_client.get_model_invocation_job.return_value = _fake_boto3_response(status="Stopped")

    batch = litellm.cancel_batch(batch_id=JOB_ARN, custom_llm_provider="bedrock")

    fake_client.stop_model_invocation_job.assert_called_once_with(jobIdentifier=JOB_ARN)
    assert batch.status == "cancelled"


class _TagGatedSTSClient:
    """Stands in for STS behind a trust policy that only admits sessions carrying ``tags``."""

    def __init__(self, tags: list[dict[str, str]], access_key_id: str) -> None:
        self._tags = tags
        self._access_key_id = access_key_id

    def get_caller_identity(self):
        return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

    def assume_role(self, **params):
        from botocore.exceptions import ClientError

        if list(params.get("Tags", ())) != self._tags:
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:TagSession"}},
                "AssumeRole",
            )
        return {
            "Credentials": {
                "AccessKeyId": self._access_key_id,
                "SecretAccessKey": "assumed-secret",
                "SessionToken": "assumed-session-token",
                "Expiration": datetime.now(timezone.utc) + timedelta(minutes=30),
            }
        }


def test_handle_model_invocation_job_status_builds_the_client_from_the_tagged_session(monkeypatch):
    """Status polling must assume the role with the deployment's session tags, like every other call."""
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]
    bedrock_client_kwargs: list[dict] = []
    fake_bedrock = MagicMock()
    fake_bedrock.get_model_invocation_job.return_value = _fake_boto3_response()

    def boto3_client(service_name, **kwargs):
        if service_name == "sts":
            return _TagGatedSTSClient(tags, "ASIABATCHSTATUSTAGGED")
        bedrock_client_kwargs.append(kwargs)
        return fake_bedrock

    with patch("boto3.client", side_effect=boto3_client):
        batch = BedrockBatchesHandler.handle_model_invocation_job_status(
            batch_id=JOB_ARN,
            aws_access_key_id="AKIABATCHSTATUSCALLER",
            aws_secret_access_key="pod-caller-secret",
            aws_role_name="arn:aws:iam::999999999999:role/litellm-batch-role",
            aws_session_name="litellm-batch-session",
            aws_session_tags=tags,
        )

    assert batch.status == "completed"
    assert [kwargs["aws_access_key_id"] for kwargs in bedrock_client_kwargs] == ["ASIABATCHSTATUSTAGGED"]


def test_cancel_batch_stops_and_polls_the_job_with_the_tagged_session(monkeypatch):
    """Cancelling on a tag-gated role must forward the deployment's session tags to both the stop and status calls."""
    import litellm

    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]
    bedrock_client_kwargs: list[dict] = []
    fake_bedrock = MagicMock()
    fake_bedrock.get_model_invocation_job.return_value = _fake_boto3_response(status="Stopped")

    def boto3_client(service_name, **kwargs):
        if service_name == "sts":
            return _TagGatedSTSClient(tags, "ASIABATCHCANCELTAGGED")
        bedrock_client_kwargs.append(kwargs)
        return fake_bedrock

    with patch("boto3.client", side_effect=boto3_client):
        batch = litellm.cancel_batch(
            batch_id=JOB_ARN,
            custom_llm_provider="bedrock",
            aws_access_key_id="AKIABATCHCANCELCALLER",
            aws_secret_access_key="pod-caller-secret",
            aws_role_name="arn:aws:iam::999999999999:role/litellm-batch-role",
            aws_session_name="litellm-batch-session",
            aws_session_tags=tags,
        )

    fake_bedrock.stop_model_invocation_job.assert_called_once_with(jobIdentifier=JOB_ARN)
    assert batch.status == "cancelled"
    assert [kwargs["aws_access_key_id"] for kwargs in bedrock_client_kwargs] == ["ASIABATCHCANCELTAGGED"] * 2


class _JsonBody:
    def __init__(self, payload: bytes) -> None:
        self._payload: Final = payload

    def stream(self) -> Iterator[bytes]:
        return iter((self._payload,))


class _AuthorizationRecorder:
    def __init__(self, body: Mapping[str, object]) -> None:
        self._payload: Final = json.dumps(body, default=str).encode()
        self.authorization_headers: tuple[str, ...] = ()

    def send(self, request: AWSPreparedRequest) -> AWSResponse:
        raw_authorization: Final = request.headers["Authorization"]
        authorization: Final = (
            raw_authorization.decode() if isinstance(raw_authorization, bytes) else str(raw_authorization)
        )
        self.authorization_headers = (*self.authorization_headers, authorization)
        return AWSResponse(request.url, 200, {"content-type": "application/json"}, _JsonBody(self._payload))


def test_retrieve_signs_with_deployment_credentials_when_env_bearer_token_is_set(monkeypatch):
    """A proxy-wide AWS_BEARER_TOKEN_BEDROCK must not override the deployment's own SigV4 credentials."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token")
    recorder: Final = _AuthorizationRecorder(_fake_boto3_response())

    with patch("botocore.httpsession.URLLib3Session.send", recorder.send):
        batch = BedrockBatchesHandler.handle_model_invocation_job_status(
            batch_id=JOB_ARN,
            aws_access_key_id="AKIADEPLOYMENTKEY",
            aws_secret_access_key="deployment-secret",
        )

    assert batch.status == "completed"
    assert len(recorder.authorization_headers) == 1
    assert recorder.authorization_headers[0].startswith("AWS4-HMAC-SHA256 Credential=AKIADEPLOYMENTKEY/")


def test_cancel_signs_with_deployment_credentials_when_env_bearer_token_is_set(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token")
    recorder: Final = _AuthorizationRecorder(_fake_boto3_response(status="Stopped"))

    with patch("botocore.httpsession.URLLib3Session.send", recorder.send):
        batch = BedrockBatchesHandler.cancel_batch(
            batch_id=JOB_ARN,
            aws_access_key_id="AKIADEPLOYMENTKEY",
            aws_secret_access_key="deployment-secret",
        )

    assert batch.status == "cancelled"
    assert len(recorder.authorization_headers) == 2
    assert all(h.startswith("AWS4-HMAC-SHA256 Credential=AKIADEPLOYMENTKEY/") for h in recorder.authorization_headers)


_BEDROCK_TEST_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    "AWS_REGION": "us-west-2",
    "AWS_DEFAULT_REGION": "us-west-2",
}


class _CaptureAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self):
        self.timeout = None
        self.event_hooks = None
        self.client_alias = "bedrock-test"
        self.put_calls = []
        self.post_calls = []
        self.batch_jobs = {}

    async def put(
        self,
        url: str,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream: bool = False,
        content=None,
    ):
        self.put_calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "params": params,
                "headers": headers or {},
                "timeout": timeout,
                "stream": stream,
                "content": content,
            }
        )
        body = data if data is not None else content
        content_bytes = body.encode("utf-8") if isinstance(body, str) else body or b""
        content_length = len(content_bytes)
        return httpx.Response(
            status_code=200,
            headers={"Content-Length": str(content_length)},
            request=httpx.Request("PUT", url),
        )

    async def post(
        self,
        url: str,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream: bool = False,
        logging_obj=None,
        files=None,
        content=None,
    ):
        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "params": params,
                "headers": headers or {},
                "timeout": timeout,
                "stream": stream,
                "content": content,
            }
        )
        raw = json if json is not None else (data if data is not None else content)
        payload = raw if isinstance(raw, dict) else json_module.loads(raw)
        job_name = payload["jobName"]
        job_arn = f"arn:aws:bedrock:us-west-2:941277531214:model-invocation-job/{job_name}"
        self.batch_jobs[job_arn] = {
            "jobArn": job_arn,
            "jobName": job_name,
            "modelId": payload["modelId"],
            "roleArn": payload["roleArn"],
            "status": "InProgress",
            "submitTime": "2026-06-02T03:50:00Z",
            "lastModifiedTime": "2026-06-02T03:55:00Z",
            "inputDataConfig": payload["inputDataConfig"],
            "outputDataConfig": payload["outputDataConfig"],
        }
        return httpx.Response(
            status_code=200,
            json={"jobArn": job_arn, "jobName": job_name, "status": "Submitted"},
            request=httpx.Request("POST", url),
        )


@pytest.mark.asyncio()
async def test_async_file_and_batch():
    """
    Test file retrieval
    """
    litellm.turn_on_debug()
    file_name = "bedrock_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    capture_client = _CaptureAsyncHTTPHandler()
    with patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV):
        with open(file_path, "rb") as batch_file:
            file_obj = await litellm.acreate_file(
                file=batch_file,
                purpose="batch",
                custom_llm_provider="bedrock",
                s3_bucket_name="litellm-proxy-941277531214",
                client=capture_client,
            )
        assert len(capture_client.put_calls) == 1
        print("CREATED FILE RESPONSE=", file_obj)

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
            return_value=capture_client,
        ):
            create_batch_response = await litellm.acreate_batch(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_obj.id,
                metadata={"key1": "value1", "key2": "value2"},
                custom_llm_provider="bedrock",
                model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                aws_batch_role_arn="arn:aws:iam::941277531214:role/service-role/AmazonBedrockExecutionRoleForAgents_BB9HNW6V4CV",
            )
            assert len(capture_client.post_calls) == 1
            print("CREATED BATCH RESPONSE=", create_batch_response)

            mock_bedrock_client = MagicMock()
            mock_bedrock_client.get_model_invocation_job.side_effect = (
                lambda jobIdentifier: capture_client.batch_jobs[jobIdentifier]
            )
            with patch("boto3.client", return_value=mock_bedrock_client):
                retrieve_batch_response = await litellm.aretrieve_batch(
                    batch_id=create_batch_response.id,
                    custom_llm_provider="bedrock",
                    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                )
            mock_bedrock_client.get_model_invocation_job.assert_called_once_with(
                jobIdentifier=create_batch_response.id
            )
            print("RETRIEVED BATCH RESPONSE=", retrieve_batch_response)

    assert retrieve_batch_response.id == create_batch_response.id
    assert retrieve_batch_response.object == "batch"
    assert retrieve_batch_response.status in [
        "validating",
        "in_progress",
        "completed",
        "failed",
        "cancelled",
    ]


@pytest.mark.asyncio()
async def test_mock_bedrock_file_url_mapping():
    """
    Simple test to capture PUT URL and validate mapping to file ID.
    """
    print("Testing Bedrock file URL mapping")

    capture_client = _CaptureAsyncHTTPHandler()
    with (
        patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV),
        open(
            os.path.join(os.path.dirname(__file__), "bedrock_batch_completions.jsonl"),
            "rb",
        ) as batch_file,
    ):
        file_obj = await litellm.acreate_file(
            file=batch_file,
            purpose="batch",
            custom_llm_provider="bedrock",
            s3_bucket_name="litellm-proxy-941277531214",
            client=capture_client,
        )

    captured_put_url = capture_client.put_calls[0]["url"]
    print(f"PUT URL: {captured_put_url}")
    print(f"File ID: {file_obj.id}")

    assert captured_put_url is not None
    assert file_obj.id.startswith("s3://")

    from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

    bedrock_config = BedrockFilesConfig()
    expected_s3_uri, _ = bedrock_config._convert_https_url_to_s3_uri(captured_put_url)
    assert file_obj.id == expected_s3_uri


@pytest.mark.asyncio()
async def test_bedrock_retrieve_batch():
    """
    Test bedrock batch retrieval functionality, validating that input and output file IDs
    are correctly extracted from the Bedrock response and included in the final transformed response.
    """
    print("Testing bedrock batch retrieval")

    mock_bedrock_response = {
        "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123",
        "jobName": "test-job-123",
        "modelId": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "roleArn": "arn:aws:iam::123456789012:role/service-role/AmazonBedrockExecutionRoleForAgents_TEST",
        "status": "Completed",
        "message": "",
        "submitTime": "2024-01-01T12:00:00Z",
        "lastModifiedTime": "2024-01-01T12:30:00Z",
        "endTime": "2024-01-01T13:00:00Z",
        "inputDataConfig": {
            "s3InputDataConfig": {"s3Uri": "s3://test-bucket/input/test-input.jsonl"}
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": "s3://test-bucket/output/"}
        },
    }

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.get_model_invocation_job.return_value = mock_bedrock_response
    mock_creds = MagicMock(access_key="ak", secret_key="sk", token="tok")

    with (
        patch("boto3.client", return_value=mock_bedrock_client),
        patch(
            "litellm.llms.bedrock.batches.transformation.BedrockBatchesConfig.get_credentials",
            return_value=mock_creds,
        ),
    ):
        batch_response = await litellm.aretrieve_batch(
            batch_id="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        assert (
            batch_response.id
            == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123"
        )
        assert batch_response.object == "batch"
        assert batch_response.status == "completed"
        assert batch_response.endpoint == "/v1/chat/completions"

        assert batch_response.input_file_id == "s3://test-bucket/input/test-input.jsonl"
        assert (
            batch_response.output_file_id
            == "s3://test-bucket/output/test-job-123/test-input.jsonl.out"
        )


def test_bedrock_batch_with_encryption_key_in_post_request():
    """
    Test that s3_encryption_key_id is included in the AWS POST request payload.
    """
    import json
    import litellm

    test_kms_key_id = (
        "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
    )

    captured_request_body = None

    def mock_post(*args, **kwargs):
        nonlocal captured_request_body
        if "data" in kwargs:
            captured_request_body = kwargs["data"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job",
            "jobName": "test-job",
            "status": "Submitted",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        return mock_response

    with (
        patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV),
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            side_effect=mock_post,
        ),
    ):
        response = litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="s3://test-bucket/input/test.jsonl",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            s3_encryption_key_id=test_kms_key_id,
            aws_batch_role_arn="arn:aws:iam::123456789012:role/test-role",
        )

    assert captured_request_body is not None, "Request body was not captured"

    request_data = json.loads(captured_request_body)
    print("REQUEST DATA to bedrock batch creation", json.dumps(request_data, indent=4))

    assert "outputDataConfig" in request_data
    assert "s3OutputDataConfig" in request_data["outputDataConfig"]
    assert "s3EncryptionKeyId" in request_data["outputDataConfig"]["s3OutputDataConfig"]
    assert (
        request_data["outputDataConfig"]["s3OutputDataConfig"]["s3EncryptionKeyId"]
        == test_kms_key_id
    )

    print("SUCCESS: s3_encryption_key_id properly included in AWS POST request")


def test_bedrock_file_upload_signing_uses_deployment_credentials(monkeypatch):
    from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

    config = BedrockFilesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, ""

    monkeypatch.setattr(config, "_sign_s3_request", capture_signing)

    result = config.transform_create_file_request(
        model="",
        create_file_data={
            "file": (
                "batch.jsonl",
                b'{"custom_id":"req-1","body":{"model":"bedrock/model"}}\n',
                "application/jsonl",
            ),
            "purpose": "batch",
        },
        optional_params={},
        litellm_params={
            "s3_bucket_name": "deployment-bucket",
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
        },
    )

    assert "eu-west-1" in result["url"]
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
    assert captured["optional_params"]["aws_secret_access_key"] == "deployment-secret"
    assert captured["optional_params"]["aws_region_name"] == "eu-west-1"


def test_bedrock_batch_signing_uses_deployment_credentials(monkeypatch):
    from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig

    config = BedrockBatchesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, b"{}"

    monkeypatch.setattr(config.common_utils, "sign_aws_request", capture_signing)

    result = config.transform_create_batch_request(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        create_batch_data={
            "input_file_id": "s3://deployment-bucket/input.jsonl",
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
        },
        optional_params={},
        litellm_params={
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
            "aws_batch_role_arn": "arn:aws:iam::123456789012:role/bedrock-batch",
        },
    )

    assert result["url"].startswith("https://bedrock.eu-west-1.amazonaws.com/")
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
    assert captured["optional_params"]["aws_secret_access_key"] == "deployment-secret"
    assert captured["optional_params"]["aws_region_name"] == "eu-west-1"


def test_bedrock_batch_retrieval_signing_uses_deployment_credentials(monkeypatch):
    from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig

    config = BedrockBatchesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, b""

    monkeypatch.setattr(config.common_utils, "sign_aws_request", capture_signing)

    result = config.transform_retrieve_batch_request(
        batch_id="arn:aws:bedrock:eu-west-1:123456789012:model-invocation-job/job-1",
        optional_params={},
        litellm_params={
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
        },
    )

    assert result["url"].startswith("https://bedrock.eu-west-1.amazonaws.com/")
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
    assert captured["optional_params"]["aws_secret_access_key"] == "deployment-secret"
    assert captured["optional_params"]["aws_region_name"] == "eu-west-1"


def test_bedrock_deployment_credentials_block_caller_profile_override(monkeypatch):
    from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig

    config = BedrockBatchesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, b"{}"

    monkeypatch.setattr(config.common_utils, "sign_aws_request", capture_signing)

    config.transform_create_batch_request(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        create_batch_data={
            "input_file_id": "s3://deployment-bucket/input.jsonl",
            "completion_window": "24h",
        },
        optional_params={"aws_profile_name": "caller-controlled-profile"},
        litellm_params={
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
            "aws_batch_role_arn": "arn:aws:iam::123456789012:role/bedrock-batch",
        },
    )

    assert "aws_profile_name" not in captured["optional_params"]
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"


def test_bedrock_file_upload_s3_region_survives_deployment_region_merge(monkeypatch):
    from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

    config = BedrockFilesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, ""

    monkeypatch.setattr(config, "_sign_s3_request", capture_signing)

    result = config.transform_create_file_request(
        model="",
        create_file_data={
            "file": (
                "batch.jsonl",
                b'{"custom_id":"req-1","body":{"model":"bedrock/model"}}\n',
                "application/jsonl",
            ),
            "purpose": "batch",
        },
        optional_params={},
        litellm_params={
            "s3_bucket_name": "deployment-bucket",
            "s3_region_name": "eu-central-1",
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "us-east-1",
        },
    )

    assert "s3.eu-central-1.amazonaws.com" in result["url"]
    assert captured["optional_params"]["aws_region_name"] == "eu-central-1"
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
