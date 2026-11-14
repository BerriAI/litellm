import asyncio
import base64
import contextlib
import copy
import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Final
from unittest.mock import patch
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import pytest
import requests
from tests.unit.integrations.s3_v3_rust_support import (
    DROP,
    Received,
    ScriptedBucket,
    log_attempt,
    log_payload,
    rust_s3_logger,
    scripted_bucket,
)

import litellm
from litellm.integrations import s3_v3
from litellm.integrations.s3_v3 import S3Logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.rust_bridge import _native
from litellm.types.integrations.s3_v2 import S3LoggerSettings
from litellm.types.integrations.custom_logger_v2 import Outcome

pytestmark = pytest.mark.skipif(not hasattr(_native, "NativeS3Log"), reason="the built _native has no NativeS3Log")

START: Final = datetime(2025, 9, 14, 9, 5, 7, 42)
_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("S3_PARTITION_GRANULARITY", "S3_LOG_PROMPTS_ONLY", "AWS_REGION", "AWS_REGION_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None)
    yield


@pytest.fixture
def bucket() -> Iterator[ScriptedBucket]:
    with scripted_bucket() as scripted:
        yield scripted


def _logger(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, **params: object
) -> S3Logger:  # kwargs-ok: S3Logger keywords
    logger: Final = rust_s3_logger(
        monkeypatch,
        **{
            "s3_bucket_name": "test-bucket",
            "s3_endpoint_url": bucket.url,
            "s3_aws_access_key_id": "test-key",
            "s3_aws_secret_access_key": "test-secret",
            "s3_region_name": "us-east-1",
            **params,
        },
    )
    _ = logger.periodic_flush_task.cancel()
    return logger


def _payload(response_id: str = "test", **fields: object) -> dict[str, object]:  # kwargs-ok: payload fields
    return {"id": response_id, "metadata": {}, "messages": [], **fields}


async def _upload(logger: S3Logger, payload: Mapping[str, object], start_time: datetime = START) -> None:
    await log_payload(logger, payload, start_time)
    await logger.flush_queue()


def _failures() -> object:
    return patch.object(s3_v3._Host, "callback_failed")


def _uploaded(put: Received) -> dict[str, object]:
    loaded: Final = json.loads(put.body)
    assert isinstance(loaded, dict)
    return loaded


def _assert_backoff_gaps(puts: list[Received]) -> None:
    gaps: Final = [later.at - earlier.at for earlier, later in zip(puts, puts[1:])]
    for attempt, gap in enumerate(gaps):
        assert (1 << attempt) * 0.5 <= gap < (1 << attempt) * 1.5 + 0.25, (attempt, gap)


class TestS3V2UnitTests:
    @pytest.mark.asyncio
    async def test_s3_v2_endpoint_url(self, monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
        logger: Final = _logger(monkeypatch, bucket)
        await _upload(logger, _payload("endpoint"))

        (put,) = bucket.puts()
        assert f"{bucket.url}{put.path}" == f"{bucket.url}/test-bucket/2025-09-14/time-09-05-07-000042_endpoint.json"
        bucket.objects["/test-bucket/2025-09-14/download-test-key.json"] = b'{"downloaded": "data"}'
        result: Final = await logger.get_proxy_server_request_from_cold_storage_with_object_key(
            "2025-09-14/download-test-key.json"
        )
        assert bucket.received[-1].path == "/test-bucket/2025-09-14/download-test-key.json"
        assert result == {"downloaded": "data"}

    @pytest.mark.asyncio
    async def test_s3_v2_put_url_encodes_spaces_in_object_key(
        self, monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
    ) -> None:
        logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=True)
        await _upload(logger, _payload("test-key", metadata={"user_api_key_team_alias": "My Team"}))

        (put,) = bucket.puts()
        s3_object_key: Final = "My Team/2025-09-14/time-09-05-07-000042_test-key.json"
        raw_url: Final = f"{bucket.url}/test-bucket/{s3_object_key}"
        expected_url: Final = requests.Request("PUT", raw_url).prepare().url
        actual_url: Final = f"{bucket.url}{put.path}"
        assert actual_url == expected_url
        assert " " not in actual_url


@pytest.mark.asyncio
async def test_async_upload_retries_on_s3_503(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    bucket.reply("test-retry", (503, ""), (200, ""))
    await _upload(_logger(monkeypatch, bucket), _payload("test-retry"))

    assert len(bucket.puts()) == 2
    _assert_backoff_gaps(bucket.puts())


@pytest.mark.asyncio
async def test_async_upload_retries_on_s3_500(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    bucket.reply("test-retry-500", (500, ""), (200, ""))
    await _upload(_logger(monkeypatch, bucket), _payload("test-retry-500"))

    assert len(bucket.puts()) == 2
    _assert_backoff_gaps(bucket.puts())


@pytest.mark.asyncio
async def test_async_upload_exhausts_retries_on_persistent_503(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("test-exhaust", (503, ""))
    logger: Final = _logger(monkeypatch, bucket)
    with _failures() as mock_failure:
        await _upload(logger, _payload("test-exhaust"))

    assert len(bucket.puts()) == 3
    _assert_backoff_gaps(bucket.puts())
    mock_failure.assert_called_once_with()


@pytest.mark.asyncio
async def test_async_upload_retries_400_with_an_unknown_error_code(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("test-retry-400", (400, "<Error><Code>SomethingElse</Code></Error>"), (200, ""))
    logger: Final = _logger(monkeypatch, bucket)
    await _upload(logger, _payload("test-retry-400"))

    assert len(bucket.puts()) == 1
    assert len(logger.sink.queued_keys()) == 1


@pytest.mark.asyncio
async def test_async_upload_is_single_attempted_on_404_through_production_http_handler(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("test-404", (404, "<Error><Code>NoSuchBucket</Code></Error>"))
    logger: Final = _logger(monkeypatch, bucket)
    await _upload(logger, _payload("test-404"))

    assert len(bucket.puts()) == 1
    assert len(logger.sink.queued_keys()) == 1


@pytest.mark.asyncio
async def test_async_upload_access_denied_403_is_retried_and_then_requeued(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("test-403-denied", (403, "<Error><Code>AccessDenied</Code></Error>"))
    logger: Final = _logger(monkeypatch, bucket)
    await _upload(logger, _payload("test-403-denied"))

    assert len(bucket.puts()) == 3
    _assert_backoff_gaps(bucket.puts())
    assert len(logger.sink.queued_keys()) == 1


@pytest.mark.asyncio
async def test_async_upload_retries_429_through_production_http_handler(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("429", (429, ""), (200, ""))
    logger: Final = _logger(monkeypatch, bucket)
    await _upload(logger, _payload("429"))

    puts: Final = bucket.puts()
    assert len(puts) == 2
    assert puts[0].body == puts[1].body
    _assert_backoff_gaps(puts)
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_async_upload_retries_transport_errors_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("transport", (DROP, ""), (200, ""))
    logger: Final = _logger(monkeypatch, bucket)
    await _upload(logger, _payload("transport"))

    assert logger.sink.queued_keys() == []
    assert len(bucket.objects) == 1


@pytest.mark.asyncio
async def test_async_upload_exhausts_transport_error_retries_and_keeps_the_event_queued(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.reply("down", (DROP, ""))
    logger: Final = _logger(monkeypatch, bucket)
    await _upload(logger, _payload("down"))

    assert logger.sink.queued_keys() == ["2025-09-14/time-09-05-07-000042_down.json"]


@pytest.mark.asyncio
async def test_async_log_event_skips_when_standard_logging_object_missing(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    kwargs_without_slo: Final = {"call_type": "afile_delete", "model": None, "litellm_call_id": "test-call-id"}

    with _failures() as mock_failure:
        await log_attempt(logger, kwargs_without_slo, START)

    assert not mock_failure.called
    assert logger.sink.queued_keys() == []


async def _stripped(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, payload: dict[str, object]
) -> dict[str, object]:
    logger: Final = _logger(monkeypatch, bucket, s3_strip_base64_files=True)
    await _upload(logger, {**_payload("strip"), **payload})
    (put,) = bucket.puts()
    return _uploaded(put)


@pytest.mark.asyncio
async def test_strip_base64_removes_file_and_nontext_entries(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    payload: Final = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "image", "file": {"file_data": "data:image/png;base64,AAAA"}},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,BBBB"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response"},
                    {"type": "audio", "file": {"file_data": "data:audio/wav;base64,CCCC"}},
                ],
            },
        ]
    }

    stripped: Final = await _stripped(monkeypatch, bucket, payload)

    messages: Final = stripped["messages"]
    assert isinstance(messages, list)
    assert len(messages[0]["content"]) == 1
    assert messages[0]["content"][0]["text"] == "Hello world"
    assert len(messages[1]["content"]) == 1
    assert messages[1]["content"][0]["text"] == "Response"
    for msg in messages:
        for content in msg["content"]:
            assert "file" not in content
            assert content.get("type") == "text"


@pytest.mark.asyncio
async def test_strip_base64_keeps_non_file_content(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    payload: Final = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Just text"}, {"type": "text", "text": "Another message"}],
            }
        ]
    }

    stripped: Final = await _stripped(monkeypatch, bucket, payload)

    assert stripped["messages"][0]["content"] == payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_strip_base64_handles_empty_or_missing_messages(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_strip_base64_files=True)
    payload_no_messages: Final = {"id": "no-messages", "metadata": {}}
    payload_empty: Final = {"id": "empty", "metadata": {}, "messages": []}
    await log_payload(logger, payload_no_messages, START)
    await log_payload(logger, payload_empty, START)
    await logger.flush_queue()

    uploaded: Final = {body["id"]: body for body in (_uploaded(put) for put in bucket.puts())}
    assert uploaded["no-messages"] == payload_no_messages
    assert uploaded["empty"] == payload_empty


@pytest.mark.asyncio
async def test_strip_base64_mixed_nested_objects(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    payload: Final = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Keep me"},
                    {"type": "custom", "metadata": "ignore but non-text"},
                    {"foo": "bar"},
                    {"file": {"file_data": "data:application/pdf;base64,XXX"}},
                ],
                "extra": {"trace_id": "123"},
            }
        ]
    }

    stripped: Final = await _stripped(monkeypatch, bucket, payload)

    content: Final = stripped["messages"][0]["content"]
    assert len(content) == 2
    assert {"type": "text", "text": "Keep me"} in content
    assert {"foo": "bar"} in content
    assert stripped["messages"][0]["extra"]["trace_id"] == "123"


@pytest.mark.asyncio
async def test_strip_base64_recursive_redaction(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    payload: Final = {
        "messages": [
            {
                "content": [
                    {"type": "text", "text": "normal text"},
                    {"type": "text", "text": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"},
                    {"type": "text", "text": "Nested: {'data': 'data:application/pdf;base64,AAA...'}"},
                    {"file": {"file_data": "data:application/pdf;base64,AAAA"}},
                    {"metadata": {"preview": "data:audio/mp3;base64,AAAAA=="}},
                ]
            }
        ]
    }

    result: Final = await _stripped(monkeypatch, bucket, payload)
    content: Final = result["messages"][0]["content"]

    assert not any("file" in c for c in content)
    for c in content:
        if isinstance(c, dict):
            s = json.dumps(c).lower()
            assert "base64," not in s, f"Found real base64 blob in: {s}"


async def _queued_key(logger: S3Logger, payload: Mapping[str, object], start_time: datetime = START) -> str:
    await log_payload(logger, payload, start_time)
    (key,) = logger.sink.queued_keys()
    return key


@pytest.mark.parametrize(
    "use_team_prefix,use_key_prefix,team_alias,key_alias,expected_prefix",
    [
        (False, False, "teamA", "keyA", ""),
        (True, False, "teamA", "keyA", "teamA/"),
        (False, True, "teamA", "keyA", "keyA/"),
        (True, True, "teamA", "keyA", "teamA/keyA/"),
        (True, True, None, "keyA", "keyA/"),
        (True, True, "teamA", None, "teamA/"),
        (True, True, None, None, ""),
    ],
)
@pytest.mark.asyncio
async def test_s3_object_key_prefix_combinations(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    use_team_prefix: bool,
    use_key_prefix: bool,
    team_alias: str | None,
    key_alias: str | None,
    expected_prefix: str,
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=use_team_prefix, s3_use_key_prefix=use_key_prefix)
    payload: Final = _payload(
        "abc123",
        metadata={"user_api_key_team_alias": team_alias, "user_api_key_alias": key_alias},
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )

    key: Final = await _queued_key(logger, payload)

    assert key == f"{expected_prefix}2025-09-14/time-09-05-07-000042_abc123.json"


@pytest.mark.asyncio
async def test_prefix_priority_and_path_construction(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=True, s3_use_key_prefix=True)
    payload: Final = _payload(
        "xyz999", metadata={"user_api_key_team_alias": "Team-Alpha", "user_api_key_alias": "API-12345"}
    )

    key: Final = await _queued_key(logger, payload)

    assert key.startswith("Team-Alpha/API-12345/2025-09-14/")
    assert "//" not in key


@pytest.mark.asyncio
async def test_prefix_absent_when_flags_disabled(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=False, s3_use_key_prefix=False)
    payload: Final = _payload(
        "no-prefix", metadata={"user_api_key_team_alias": "team-x", "user_api_key_alias": "key-x"}
    )

    key: Final = await _queued_key(logger, payload)

    assert key == "2025-09-14/time-09-05-07-000042_no-prefix.json"


@pytest.mark.asyncio
async def test_combined_prefix_reflects_in_s3_object_key(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=True, s3_use_key_prefix=True)
    payload: Final = _payload(
        "int-test", metadata={"user_api_key_team_alias": "myteam", "user_api_key_alias": "apikey"}
    )

    key: Final = await _queued_key(logger, payload)

    assert "myteam/apikey/" in key, f"Expected both prefixes in key: {key}"


@pytest.mark.asyncio
async def test_s3_object_key_sanitizes_slashes_and_colons_in_file_name(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_path="LiteLLMAPPLogs", s3_use_team_prefix=True)
    payload: Final = _payload(
        "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/gl18r6skk9yy",
        metadata={"user_api_key_team_alias": "myteam"},
    )

    key: Final = await _queued_key(logger, payload, datetime(2026, 2, 11, 0, 35, 18, 391582))

    assert key == (
        "LiteLLMAPPLogs/myteam/2026-02-11/"
        "time-00-35-18-391582_arn_aws_bedrock_us-east-1_123456789012_model-invocation-job_gl18r6skk9yy.json"
    )


@pytest.mark.parametrize(
    "response_id",
    [
        "s3://example-batch-bucket/litellm-bedrock-files/input.jsonl",
        "gs://example-batch-bucket/litellm-vertex-files/input.jsonl",
    ],
)
@pytest.mark.asyncio
async def test_s3_object_key_has_no_colon_for_cloud_uri_file_ids(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, response_id: str
) -> None:
    logger: Final = _logger(monkeypatch, bucket)

    key: Final = await _queued_key(logger, _payload(response_id), datetime(2026, 9, 7, 4, 51, 6, 685889))

    filename: Final = key.rsplit("/", 1)[-1]
    assert ":" not in filename
    assert filename.endswith("_input.jsonl.json")


@pytest.mark.asyncio
async def test_create_s3_batch_logging_element_flat_key_for_arn_response_id(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=False, s3_use_key_prefix=False)
    payload: Final = _payload("arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/gl18r6skk9yy")

    key: Final = await _queued_key(logger, payload, datetime(2026, 2, 11, 0, 35, 18, 391582))

    file_segment: Final = key.split("2026-02-11/", 1)[1]
    assert "/" not in file_segment, f"Expected flat file under date segment, got: {key}"
    assert file_segment.endswith("model-invocation-job_gl18r6skk9yy.json")


def _expected_content_md5(payload: dict[str, object]) -> str:
    json_string: Final = safe_dumps(payload)
    return base64.b64encode(hashlib.md5(json_string.encode("utf-8"), usedforsecurity=False).digest()).decode()


@pytest.mark.asyncio
async def test_async_upload_sets_content_md5_header(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    payload: Final = _payload("test-md5", test="content-md5")
    await _upload(_logger(monkeypatch, bucket), payload)

    (put,) = bucket.puts()
    assert put.headers["content-md5"] == _expected_content_md5(payload)
    assert "x-amz-server-side-encryption" not in put.headers


@pytest.mark.asyncio
async def test_async_upload_sets_server_side_encryption_header_when_configured(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    await _upload(_logger(monkeypatch, bucket, s3_server_side_encryption="aws:kms"), _payload("test-sse"))

    (put,) = bucket.puts()
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"


async def _global_params_put(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, params: Mapping[str, object]
) -> Received:
    monkeypatch.setattr(
        litellm,
        "s3_callback_params",
        {
            "s3_endpoint_url": bucket.url,
            "s3_region_name": "us-east-1",
            "s3_aws_access_key_id": "test-key",
            "s3_aws_secret_access_key": "test-secret",
            **params,
        },
    )
    logger: Final = rust_s3_logger(monkeypatch)
    await _upload(logger, _payload("global"))
    (put,) = bucket.puts()
    return put


@pytest.mark.asyncio
async def test_s3_server_side_encryption_read_from_callback_params(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    put: Final = await _global_params_put(
        monkeypatch, bucket, {"s3_bucket_name": "from-global", "s3_server_side_encryption": "aws:kms"}
    )
    assert put.path.startswith("/from-global/")
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"


@pytest.mark.asyncio
async def test_async_upload_sets_sse_kms_key_id_header_when_configured(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(
        monkeypatch,
        bucket,
        s3_server_side_encryption="aws:kms",
        s3_sse_kms_key_id="arn:aws:kms:us-east-1:111122223333:key/test-key-id",
    )
    await _upload(logger, _payload("test-kms"))

    (put,) = bucket.puts()
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"
    assert put.headers["x-amz-server-side-encryption-aws-kms-key-id"] == (
        "arn:aws:kms:us-east-1:111122223333:key/test-key-id"
    )


@pytest.mark.asyncio
async def test_async_upload_omits_kms_key_id_header_when_not_configured(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    await _upload(_logger(monkeypatch, bucket, s3_server_side_encryption="aws:kms"), _payload("test-no-kms"))

    (put,) = bucket.puts()
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"
    assert "x-amz-server-side-encryption-aws-kms-key-id" not in put.headers


@pytest.mark.asyncio
async def test_s3_sse_kms_key_id_read_from_callback_params(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    put: Final = await _global_params_put(
        monkeypatch,
        bucket,
        {
            "s3_bucket_name": "from-global",
            "s3_server_side_encryption": "aws:kms",
            "s3_sse_kms_key_id": "arn:aws:kms:us-east-1:111122223333:key/test-key-id",
        },
    )
    assert put.headers["x-amz-server-side-encryption-aws-kms-key-id"] == (
        "arn:aws:kms:us-east-1:111122223333:key/test-key-id"
    )


@pytest.mark.asyncio
async def test_async_upload_infers_aws_kms_when_only_key_id_set(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_sse_kms_key_id="arn:aws:kms:us-east-1:111122223333:key/test-key-id")
    await _upload(logger, _payload("test-kms-only"))

    (put,) = bucket.puts()
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"
    assert put.headers["x-amz-server-side-encryption-aws-kms-key-id"] == (
        "arn:aws:kms:us-east-1:111122223333:key/test-key-id"
    )


@pytest.mark.asyncio
async def test_s3_sse_kms_key_id_read_from_audit_override_params(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setattr(litellm, "s3_callback_params", {"s3_bucket_name": "normal-logs-bucket"})
    logger: Final = rust_s3_logger(
        monkeypatch,
        s3_bucket_name="audit-logs-bucket",
        s3_sse_kms_key_id="arn:aws:kms:us-east-1:111122223333:key/audit-key-id",
        s3_endpoint_url=bucket.url,
        s3_region_name="us-east-1",
        s3_aws_access_key_id="test-key",
        s3_aws_secret_access_key="test-secret",
    )
    await _upload(logger, _payload("audit"))

    (put,) = bucket.puts()
    assert put.path.startswith("/audit-logs-bucket/")
    assert put.headers["x-amz-server-side-encryption-aws-kms-key-id"] == (
        "arn:aws:kms:us-east-1:111122223333:key/audit-key-id"
    )


@pytest.mark.asyncio
async def test_kms_key_id_dropped_when_algorithm_is_not_kms(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    put: Final = await _global_params_put(
        monkeypatch,
        bucket,
        {
            "s3_bucket_name": "from-global",
            "s3_server_side_encryption": "AES256",
            "s3_sse_kms_key_id": "arn:aws:kms:us-east-1:111122223333:key/test-key-id",
        },
    )
    assert put.headers["x-amz-server-side-encryption"] == "AES256"
    assert "x-amz-server-side-encryption-aws-kms-key-id" not in put.headers


@pytest.mark.asyncio
async def test_non_string_algorithm_is_dropped_and_valid_key_id_is_rescued(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    put: Final = await _global_params_put(
        monkeypatch,
        bucket,
        {
            "s3_bucket_name": "from-global",
            "s3_server_side_encryption": True,
            "s3_sse_kms_key_id": "arn:aws:kms:us-east-1:111122223333:key/test-key-id",
        },
    )
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"
    assert put.headers["x-amz-server-side-encryption-aws-kms-key-id"] == (
        "arn:aws:kms:us-east-1:111122223333:key/test-key-id"
    )


@pytest.mark.asyncio
async def test_non_string_key_id_is_dropped_and_valid_algorithm_is_kept(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    put: Final = await _global_params_put(
        monkeypatch,
        bucket,
        {"s3_bucket_name": "from-global", "s3_server_side_encryption": "aws:kms", "s3_sse_kms_key_id": 12345},
    )
    assert put.headers["x-amz-server-side-encryption"] == "aws:kms"
    assert "x-amz-server-side-encryption-aws-kms-key-id" not in put.headers


def _signature_for(signer_cls: type, url: str, method: str, body: bytes | None, headers: Mapping[str, str]) -> str:
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    signed_header_names: Final = headers["authorization"].split("SignedHeaders=")[1].split(", ")[0]
    request: Final = AWSRequest(
        method=method,
        url=url,
        data=body,
        headers={name: headers[name] for name in signed_header_names.split(";") if name in headers},
    )
    request.context["timestamp"] = headers["x-amz-date"]
    signer: Final = signer_cls(Credentials(_ACCESS_KEY, _SECRET_KEY), "s3", "us-east-1")
    canonical_request: Final = signer.canonical_request(request)
    return signer.signature(signer.string_to_sign(request, canonical_request), request)


def _as_s3_canonicalizes(url: str) -> str:
    split: Final = urlsplit(url)
    return urlunsplit(split._replace(path=quote(unquote(split.path), safe="/~")))


def _assert_signed_for_s3_canonicalization(
    url: str, method: str, body: bytes | None, headers: Mapping[str, str]
) -> None:
    from botocore.auth import S3SigV4Auth, SigV4Auth

    assert url == _as_s3_canonicalizes(url)
    sent_signature: Final = headers["authorization"].split("Signature=")[1].strip()
    assert sent_signature == _signature_for(S3SigV4Auth, url, method, body, headers)
    assert sent_signature != _signature_for(SigV4Auth, url, method, body, headers)


def _logger_for_signing(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, **params: object
) -> S3Logger:  # kwargs-ok: S3Logger keywords
    return rust_s3_logger(
        monkeypatch,
        s3_bucket_name="logs-bucket",
        s3_endpoint_url=bucket.url,
        s3_aws_access_key_id=_ACCESS_KEY,
        s3_aws_secret_access_key=_SECRET_KEY,
        s3_region_name="us-east-1",
        **params,
    )


_KEY_WITH_SPACE = "LOGS/LLM AI Projects/2026-08-04/time-13-01-00-abc.json"
_RESERVED_CHAR_KEYS = (
    "2026-08-21/time-05-29-36_resp_bGl0ZWxsbTpjdXN0b20=.json",
    "session=logs/2026-08-21/time-05-29-36_abc.json",
    "a+b/2026-08-21/time-05-29-36_abc.json",
    "a&b/2026-08-21/time-05-29-36_abc.json",
    "a#b/2026-08-21/time-05-29-36_abc.json",
    "a?b/2026-08-21/time-05-29-36_abc.json",
    "a%b/2026-08-21/time-05-29-36_abc.json",
    _KEY_WITH_SPACE,
)


@pytest.mark.asyncio
async def test_async_upload_signs_object_key_with_space_the_way_s3_does(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger_for_signing(monkeypatch, bucket, s3_path="LOGS", s3_use_team_prefix=True)
    await _upload(
        logger, _payload("abc", metadata={"user_api_key_team_alias": "LLM AI Projects"}), datetime(2026, 8, 4, 13, 1)
    )

    (put,) = bucket.puts()
    assert unquote(put.path) == "/logs-bucket/LOGS/LLM AI Projects/2026-08-04/time-13-01-00-000000_abc.json"
    _assert_signed_for_s3_canonicalization(f"{bucket.url}{put.path}", "PUT", put.body, put.headers)


@pytest.mark.asyncio
async def test_download_signs_object_key_with_space_the_way_s3_does(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    bucket.objects[f"/logs-bucket/{quote(_KEY_WITH_SPACE)}"] = b'{"downloaded": "data"}'
    logger: Final = _logger_for_signing(monkeypatch, bucket)

    assert await logger.get_proxy_server_request_from_cold_storage_with_object_key(_KEY_WITH_SPACE) == {
        "downloaded": "data"
    }

    get: Final = bucket.received[-1]
    _assert_signed_for_s3_canonicalization(f"{bucket.url}{get.path}", "GET", None, get.headers)


@pytest.mark.parametrize("s3_object_key", _RESERVED_CHAR_KEYS)
@pytest.mark.asyncio
async def test_download_percent_encodes_reserved_characters_in_object_key(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, s3_object_key: str
) -> None:
    bucket.objects[f"/logs-bucket/{quote(s3_object_key, safe='/')}"] = b'{"downloaded": "data"}'
    logger: Final = _logger_for_signing(monkeypatch, bucket)

    assert await logger.get_proxy_server_request_from_cold_storage_with_object_key(s3_object_key) == {
        "downloaded": "data"
    }

    get: Final = bucket.received[-1]
    assert get.path == f"/logs-bucket/{quote(s3_object_key, safe='/')}"
    _assert_signed_for_s3_canonicalization(f"{bucket.url}{get.path}", "GET", None, get.headers)


def _chat_payload() -> dict[str, object]:
    return {
        "id": "chatcmpl-prompts-only",
        "messages": [{"role": "user", "content": "Reply with exactly the word PINEAPPLE."}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "PINEAPPLE"}}]},
        "metadata": {"user_api_key_team_alias": None},
    }


async def _uploaded_chat_body(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, params: Mapping[str, object], outcome: Outcome
) -> dict[str, object]:
    monkeypatch.setattr(
        litellm,
        "s3_callback_params",
        {
            "s3_bucket_name": "test-bucket",
            "s3_endpoint_url": bucket.url,
            "s3_aws_access_key_id": "test-key",
            "s3_aws_secret_access_key": "test-secret",
            "s3_region_name": "us-east-1",
            **params,
        },
    )
    logger: Final = rust_s3_logger(monkeypatch)
    payload: Final = _chat_payload()
    original: Final = copy.deepcopy(payload)
    await log_attempt(logger, {"standard_logging_object": payload}, datetime(2026, 7, 30, 12, 0, 0), outcome)
    assert payload == original, "the caller's standard_logging_object must not be mutated"
    await logger.flush_queue()
    (put,) = bucket.puts()
    return _uploaded(put)


@pytest.mark.asyncio
@pytest.mark.parametrize("event_name", ["async_log_success_event", "async_log_failure_event"])
async def test_prompts_only_drops_response_but_keeps_messages_in_uploaded_object(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, event_name: str
) -> None:
    outcome: Final = Outcome.success if event_name == "async_log_success_event" else Outcome.failure
    body: Final = await _uploaded_chat_body(monkeypatch, bucket, {"s3_log_prompts_only": True}, outcome)

    assert body["messages"] == _chat_payload()["messages"]
    assert body["response"] is None
    assert body["id"] == "chatcmpl-prompts-only"


@pytest.mark.asyncio
async def test_prompts_only_default_off_keeps_response_in_uploaded_object(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    body: Final = await _uploaded_chat_body(monkeypatch, bucket, {}, Outcome.success)

    assert body["response"] == _chat_payload()["response"]
    assert body["messages"] == _chat_payload()["messages"]


@pytest.mark.asyncio
async def test_prompts_only_explicit_false_in_params_beats_env_var(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setenv("S3_LOG_PROMPTS_ONLY", "true")
    body: Final = await _uploaded_chat_body(monkeypatch, bucket, {"s3_log_prompts_only": False}, Outcome.success)

    assert body["response"] == _chat_payload()["response"]


@pytest.mark.asyncio
async def test_prompts_only_env_var_applies_when_param_unset(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setenv("S3_LOG_PROMPTS_ONLY", "true")
    body: Final = await _uploaded_chat_body(monkeypatch, bucket, {}, Outcome.success)

    assert body["response"] is None


_PARTITION_START: Final = datetime(2026, 9, 29, 14, 5, 9, 123456)
_PARTITION_ID: Final = "chatcmpl-partition"
_DAILY_KEY: Final = f"logs/team-a/key-a/2026-09-29/time-14-05-09-123456_{_PARTITION_ID}.json"
_HOURLY_KEY: Final = f"logs/team-a/key-a/2026-09-29/14/time-14-05-09-123456_{_PARTITION_ID}.json"


def _partition_payload(response_id: str = _PARTITION_ID) -> dict[str, object]:
    return _payload(response_id, metadata={"user_api_key_team_alias": "team-a", "user_api_key_alias": "key-a"})


def _partition_logger(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, callback_params: Mapping[str, object]
) -> S3Logger:
    monkeypatch.setattr(
        litellm,
        "s3_callback_params",
        {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_path": "logs",
            "s3_endpoint_url": bucket.url,
            "s3_aws_access_key_id": "test-key",
            "s3_aws_secret_access_key": "test-secret",
            "s3_use_team_prefix": True,
            "s3_use_key_prefix": True,
            **callback_params,
        },
    )
    return rust_s3_logger(monkeypatch)


@pytest.mark.parametrize(
    ("callback_params", "expected_key"),
    [
        ({}, _DAILY_KEY),
        ({"s3_partition_granularity": None}, _DAILY_KEY),
        ({"s3_partition_granularity": "day"}, _DAILY_KEY),
        ({"s3_partition_granularity": "hour"}, _HOURLY_KEY),
    ],
)
@pytest.mark.asyncio
async def test_partition_granularity_sets_request_log_folder(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, callback_params: Mapping[str, object], expected_key: str
) -> None:
    logger: Final = _partition_logger(monkeypatch, bucket, callback_params)

    assert await _queued_key(logger, _partition_payload(), _PARTITION_START) == expected_key


@pytest.mark.parametrize("invalid", ["hourly", "HOUR", "1", 1, True])
@pytest.mark.asyncio
async def test_invalid_partition_granularity_warns_and_keeps_daily_folder(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, invalid: object
) -> None:
    logger: Final = _partition_logger(monkeypatch, bucket, {"s3_partition_granularity": invalid})
    await log_payload(logger, _partition_payload(), _PARTITION_START)
    await log_payload(logger, _partition_payload(), _PARTITION_START)

    assert logger.sink.queued_keys() == [_DAILY_KEY, _DAILY_KEY]


@pytest.mark.asyncio
async def test_partition_granularity_reads_admin_ui_env_var_below_callback_params(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setenv("S3_PARTITION_GRANULARITY", "hour")

    from_env: Final = await _queued_key(
        _partition_logger(monkeypatch, bucket, {}), _partition_payload(), _PARTITION_START
    )
    from_params: Final = await _queued_key(
        _partition_logger(monkeypatch, bucket, {"s3_partition_granularity": "day"}),
        _partition_payload(),
        _PARTITION_START,
    )

    assert from_env == _HOURLY_KEY
    assert from_params == _DAILY_KEY


@pytest.mark.asyncio
async def test_partition_granularity_constructor_argument_and_os_environ_reference(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setenv("MY_S3_PARTITION", "hour")

    from_secret: Final = _partition_logger(
        monkeypatch, bucket, {"s3_partition_granularity": "os.environ/MY_S3_PARTITION"}
    )

    assert await _queued_key(from_secret, _partition_payload(), _PARTITION_START) == _HOURLY_KEY


@pytest.mark.asyncio
async def test_hourly_partition_long_key_keeps_hour_folder_within_s3_limit(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    logger: Final = _partition_logger(monkeypatch, bucket, {"s3_partition_granularity": "hour", "s3_path": "p" * 1100})

    key: Final = await _queued_key(logger, _partition_payload("r" * 600), _PARTITION_START)

    assert len(key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert re.search(r"/2026-09-29/14/[0-9a-f]{64}\.json$", key)


@pytest.mark.asyncio
@pytest.mark.parametrize(("granularity", "hour_folder"), [("hour", True), ("day", False), (None, False)])
async def test_audit_log_key_follows_audit_callback_params_partition_granularity(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, granularity: str | None, hour_folder: bool
) -> None:
    logger: Final = rust_s3_logger(
        monkeypatch,
        s3_bucket_name="audit-bucket",
        s3_path="audit",
        s3_partition_granularity=granularity,
        s3_endpoint_url=bucket.url,
    )

    await logger.async_log_audit_log_event({"id": "audit-1"})

    (key,) = logger.sink.queued_keys()
    match: Final = re.fullmatch(
        r"audit/audit_logs/\d{4}-\d{2}-\d{2}/(?:(\d{2})/)?(\d{2})-\d{2}-\d{2}_audit-1\.json", key
    )
    assert match is not None, key
    assert (match.group(1) is not None) is hour_folder
    if hour_folder:
        assert match.group(1) == match.group(2)


@pytest.mark.asyncio
async def test_hourly_batch_file_upload_writes_one_file_per_hour_folder(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _partition_logger(
        monkeypatch, bucket, {"s3_partition_granularity": "hour", "s3_batch_file_upload": True}
    )
    await log_payload(logger, _partition_payload("before"), datetime(2026, 9, 29, 13, 59, 59))
    await log_payload(logger, _partition_payload("after"), datetime(2026, 9, 29, 14, 0, 1))

    await logger.flush_queue()

    by_folder: Final = {
        re.sub(
            r"/batch_\d{2}-\d{2}-\d{2}_[0-9a-f]{32}\.jsonl$", "", unquote(put.path).removeprefix("/test-bucket/")
        ): put.body
        for put in bucket.puts()
    }
    assert sorted(by_folder) == ["logs/team-a/key-a/2026-09-29/13", "logs/team-a/key-a/2026-09-29/14"]
    assert [json.loads(line)["id"] for line in by_folder["logs/team-a/key-a/2026-09-29/13"].splitlines()] == ["before"]
    assert [json.loads(line)["id"] for line in by_folder["logs/team-a/key-a/2026-09-29/14"].splitlines()] == ["after"]


def _cold_storage_key() -> str:
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    return StandardLoggingPayloadSetup._generate_cold_storage_object_key(
        start_time=_PARTITION_START, response_id=_PARTITION_ID
    )


def _cold_logger(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, params: Mapping[str, object]) -> S3Logger:
    monkeypatch.setattr(
        litellm, "s3_callback_params", {"s3_bucket_name": "test-bucket", "s3_endpoint_url": bucket.url, **params}
    )
    return rust_s3_logger(monkeypatch)


@pytest.mark.parametrize("granularity", [None, "day", "hour"])
@pytest.mark.asyncio
async def test_cold_storage_object_key_matches_the_uploaded_request_log_key(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, granularity: str | None
) -> None:
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    logger: Final = _cold_logger(monkeypatch, bucket, {"s3_path": "coldlogs", "s3_partition_granularity": granularity})
    uploaded: Final = await _queued_key(logger, _payload(_PARTITION_ID), _PARTITION_START)

    monkeypatch.setattr(litellm, "callbacks", [logger])
    cold_key: Final = _cold_storage_key()

    assert cold_key == uploaded
    assert ("/2026-09-29/14/" in cold_key) is (granularity == "hour")


@pytest.mark.asyncio
async def test_cold_storage_key_matches_upload_when_env_var_changes_mid_request(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    logger: Final = _cold_logger(monkeypatch, bucket, {"s3_path": "coldlogs"})
    monkeypatch.setattr(litellm, "callbacks", [logger])

    cold_key: Final = _cold_storage_key()
    monkeypatch.setenv("S3_PARTITION_GRANULARITY", "hour")
    uploaded: Final = await _queued_key(
        logger, _payload(_PARTITION_ID, metadata={"cold_storage_object_key": cold_key}), _PARTITION_START
    )

    assert cold_key == uploaded == f"coldlogs/2026-09-29/time-14-05-09-123456_{_PARTITION_ID}.json"


@pytest.mark.asyncio
async def test_hour_upload_ignores_a_cold_storage_key_owned_by_another_logger(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "gcs_bucket")
    logger: Final = _cold_logger(monkeypatch, bucket, {"s3_partition_granularity": "hour"})
    cold_key: Final = _cold_storage_key()
    uploaded: Final = await _queued_key(
        logger, _payload(_PARTITION_ID, metadata={"cold_storage_object_key": cold_key}), _PARTITION_START
    )

    assert cold_key == f"2026-09-29/time-14-05-09-123456_{_PARTITION_ID}.json"
    assert uploaded == f"2026-09-29/14/time-14-05-09-123456_{_PARTITION_ID}.json"


def _key(response_id: str) -> str:
    return f"2025-09-14/time-09-05-07-000042_{response_id}.json"


async def _log_all(logger: S3Logger, payloads: list[dict[str, object]]) -> None:
    for payload in payloads:
        await log_payload(logger, payload, START)


async def _log_once_a_put_arrives(logger: S3Logger, bucket: ScriptedBucket, payloads: list[dict[str, object]]) -> None:
    while not bucket.puts():
        await asyncio.sleep(0.005)
    await _log_all(logger, payloads)


def _put_calls(bucket: ScriptedBucket) -> list[tuple[str, bytes]]:
    return [(put.path, put.body) for put in bucket.puts()]


@pytest.mark.asyncio
async def test_async_send_batch_bounds_concurrent_uploads(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads=4)
    bucket.delay = 0.05
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(40)])

    await logger.flush_queue()

    assert bucket.peak == 4
    assert len(bucket.puts()) == 40


@pytest.mark.asyncio
async def test_async_send_batch_uploads_single_jsonl_file(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_batch_file_upload=True)
    payloads: Final[list[dict[str, object]]] = [{"id": "req-1"}, {"id": "req-2"}, {"id": "req-3"}]
    await _log_all(logger, payloads)

    await logger.flush_queue()

    (put,) = bucket.puts()
    assert put.path.endswith(".jsonl")
    assert [json.loads(line) for line in put.body.decode().splitlines()] == payloads
    assert put.headers["content-type"] == "application/x-ndjson"


@pytest.mark.asyncio
async def test_flush_queue_preserves_events_added_during_upload(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.delay = 0.2
    await log_payload(logger, _payload("first"), START)

    await asyncio.gather(logger.flush_queue(), _log_once_a_put_arrives(logger, bucket, [_payload("late")]))

    assert logger.sink.queued_keys() == [_key("late")]


@pytest.mark.parametrize(("configured", "batch_file"), [("false", False), ("true", True)])
@pytest.mark.asyncio
async def test_env_backed_false_string_keeps_per_request_uploads(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, configured: str, batch_file: bool
) -> None:
    await _upload(_logger(monkeypatch, bucket, s3_batch_file_upload=configured), _payload("req-1"))

    (put,) = bucket.puts()
    assert put.path.endswith(".jsonl") is batch_file


@pytest.mark.parametrize("bad", [0, -3, "0", "abc", ""])
@pytest.mark.asyncio
async def test_invalid_concurrency_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, bad: object
) -> None:
    from litellm.constants import DEFAULT_S3_MAX_CONCURRENT_UPLOADS

    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads=bad)
    bucket.delay = 0.1
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(2 * DEFAULT_S3_MAX_CONCURRENT_UPLOADS)])

    await logger.flush_queue()

    assert bucket.peak == DEFAULT_S3_MAX_CONCURRENT_UPLOADS


@pytest.mark.asyncio
async def test_env_backed_concurrency_string_is_parsed(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads="4")
    bucket.delay = 0.05
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(12)])

    await logger.flush_queue()

    assert bucket.peak == 4


@pytest.mark.asyncio
async def test_failed_uploads_stay_queued_for_next_flush(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("_elem-2.json", (503, "<Error><Code>SlowDown</Code></Error>"))
    bucket.reply("_elem-4.json", (503, "<Error><Code>SlowDown</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    await logger.flush_queue()

    assert logger.sink.queued_keys() == [_key("elem-2"), _key("elem-4")]

    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_batch_file_upload_failure_keeps_whole_batch(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_batch_file_upload=True)
    bucket.reply(".jsonl", (503, "<Error><Code>SlowDown</Code></Error>"))
    payloads: Final[list[dict[str, object]]] = [{"id": f"elem-{i}", "i": i} for i in range(3)]
    await _log_all(logger, payloads)

    await logger.flush_queue()

    assert len(bucket.puts()) == 3
    assert len(logger.sink.queued_keys()) == 1
    assert bucket.puts()[-1].body.decode() == "\n".join(json.dumps(payload) for payload in payloads)


@pytest.mark.asyncio
async def test_events_appended_during_failed_flush_survive(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("_first.json", (503, "<Error><Code>SlowDown</Code></Error>"))
    await log_payload(logger, _payload("first"), START)

    await asyncio.gather(logger.flush_queue(), _log_once_a_put_arrives(logger, bucket, [_payload("late")]))

    assert logger.sink.queued_keys() == [_key("first"), _key("late")]

    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_batch_file_key_shape(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    await _upload(_logger(monkeypatch, bucket, s3_path="logs", s3_batch_file_upload=True), {"id": "req-1"})

    (put,) = bucket.puts()
    assert re.search(r".*/2025-09-14/batch_\d{2}-\d{2}-\d{2}_[0-9a-f]{32}\.jsonl$", put.path)
    assert put.headers["content-disposition"].endswith('.jsonl"')


@pytest.mark.asyncio
async def test_failed_batch_file_is_requeued_and_resent_unchanged(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_batch_file_upload=True)
    bucket.reply(".jsonl", (503, "<Error><Code>SlowDown</Code></Error>"))
    await _log_all(logger, [{"id": f"elem-{i}", "i": i} for i in range(3)])

    await logger.flush_queue()

    (queued,) = logger.sink.queued_keys()
    assert queued.endswith(".jsonl")

    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []
    calls: Final = _put_calls(bucket)
    assert len(calls) == 4
    assert calls[0] == calls[1]


@pytest.mark.asyncio
async def test_elements_appended_after_failed_batch_file_get_their_own_file(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_batch_file_upload=True)
    bucket.reply(".jsonl", (503, "<Error><Code>SlowDown</Code></Error>"))
    await log_payload(logger, {"id": "first"}, START)

    await logger.flush_queue()

    await log_payload(logger, {"id": "late"}, START)
    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []
    calls: Final = _put_calls(bucket)
    assert len(calls) == 5
    assert calls[0] == calls[1]
    second_flush: Final = calls[3:]
    assert calls[0] in second_flush
    late_call: Final = next(call for call in second_flush if call != calls[0])
    assert late_call[0] != calls[0][0]
    assert late_call[1].decode() == json.dumps({"id": "late"})


@pytest.mark.asyncio
async def test_batch_file_mode_disabled_when_s3_v2_is_cold_storage_logger(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_batch_file_upload=True)

    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    await _upload(logger, {"id": "req-1"})

    assert len(bucket.puts()) == 1
    assert bucket.puts()[0].path.endswith("_req-1.json")

    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None)
    await _upload(logger, {"id": "req-2"})

    assert len(bucket.puts()) == 2
    assert bucket.puts()[1].path.endswith(".jsonl")


def _coded_body(code: str | None, raw_body: str | None = None) -> str:
    return raw_body if raw_body is not None else (f"<Error><Code>{code}</Code></Error>" if code is not None else "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "raw_body", "puts_per_element"),
    [
        pytest.param(403, "AccessDenied", None, 3, id="access-denied-403"),
        pytest.param(403, None, None, 3, id="empty-403"),
        pytest.param(403, None, "<html>Forbidden</html>", 3, id="html-403"),
        pytest.param(400, "KMS.DisabledException", None, 1, id="kms-disabled-400"),
        pytest.param(404, "NoSuchBucket", None, 1, id="no-such-bucket-404"),
    ],
)
async def test_non_terminal_failure_is_requeued_and_delivered_on_recovery(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    status: int,
    code: str | None,
    raw_body: str | None,
    puts_per_element: int,
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("/test-bucket/", (status, _coded_body(code, raw_body)))
    keys: Final = [_key(f"elem-{i}") for i in range(5)]
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    await logger.flush_queue()

    assert len(logger.sink.queued_keys()) == 5
    assert len(bucket.puts()) == 5 * puts_per_element

    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []
    assert len(bucket.puts()) == 5 * puts_per_element + 5
    landed: Final = frozenset(key for key in keys if any(put.path.endswith(key) for put in bucket.puts()[-5:]))
    assert landed == frozenset(keys)


@pytest.mark.asyncio
async def test_persistent_500_stays_queued_through_a_dozen_failed_flushes(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("/test-bucket/", (500, "<Error><Code>InternalError</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    for _ in range(12):
        await logger.flush_queue()
        assert len(logger.sink.queued_keys()) == 5

    assert len(bucket.puts()) == 12 * 15

    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []
    assert len(bucket.puts()) == 12 * 15 + 5


@pytest.mark.asyncio
async def test_terminal_object_is_dropped_once_next_to_delivered_siblings_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_drop_on_terminal_error=True)
    bucket.reply("_elem-1.json", (400, "<Error><Code>EntityTooLarge</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    await logger.flush_queue()

    assert len(bucket.puts()) == 5
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_terminal_object_is_requeued_when_opted_out(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_drop_on_terminal_error=False)
    bucket.reply("_elem-1.json", (400, "<Error><Code>EntityTooLarge</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    await logger.flush_queue()

    (queued,) = logger.sink.queued_keys()
    assert queued.endswith("_elem-1.json")
    assert sum(put.path.endswith("_elem-1.json") for put in bucket.puts()) == 1
    assert len(bucket.puts()) == 5


@pytest.mark.asyncio
async def test_terminal_objects_are_requeued_when_every_upload_in_the_flush_fails(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_drop_on_terminal_error=True)
    bucket.reply("/test-bucket/", (400, "<Error><Code>EntityTooLarge</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    await logger.flush_queue()

    assert len(logger.sink.queued_keys()) == 5


@pytest.mark.asyncio
async def test_overflow_after_a_failed_flush_trims_failed_first_and_counts_upload_failures_only(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_queue_size=4)
    bucket.reply("/test-bucket/", (503, "<Error><Code>SlowDown</Code></Error>"))
    await _log_all(logger, [_payload("first"), _payload("second")])

    with _failures() as mock_failure:
        await asyncio.gather(
            logger.flush_queue(),
            _log_once_a_put_arrives(logger, bucket, [_payload(f"late-{index}") for index in range(3)]),
        )

    assert logger.sink.queued_keys() == [_key("second"), _key("late-0"), _key("late-1"), _key("late-2")]
    failed_uploads: Final = 2
    assert mock_failure.call_count == failed_uploads


@pytest.mark.asyncio
async def test_queue_grows_past_the_cap_while_the_sink_fails_and_everything_lands(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_queue_size=5)
    bucket.reply("/test-bucket/", (503, "<Error><Code>SlowDown</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{index}") for index in range(5)])

    with _failures() as mock_failure:
        await logger.flush_queue()

    assert len(logger.sink.queued_keys()) == 5
    upload_failures: Final = 5
    assert mock_failure.call_count == upload_failures

    await _log_all(logger, [_payload(f"elem-{index}") for index in range(5, 8)])

    bucket.script.clear()
    await logger.flush_queue()

    assert logger.sink.queued_keys() == []
    landed: Final = frozenset(put.path.rsplit("_", 1)[-1] for put in bucket.puts()[-8:])
    assert landed == frozenset(f"elem-{index}.json" for index in range(8))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        pytest.param(404, "NoSuchKey", id="404"),
        pytest.param(401, None, id="401"),
        pytest.param(400, None, id="uncoded-400"),
    ],
)
async def test_unlisted_status_gets_one_put_and_stays_queued(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, status: int, code: str | None
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("/test-bucket/", (status, _coded_body(code)))
    await _log_all(logger, [_payload(f"elem-{index}") for index in range(4)])

    started: Final = time.monotonic()
    await logger.flush_queue()

    assert len(bucket.puts()) == 4
    assert time.monotonic() - started < 0.5
    assert len(logger.sink.queued_keys()) == 4


_SLOW_DOWN: Final = (503, "<Error><Code>SlowDown</Code></Error>")


def _names(puts: list[Received]) -> list[str]:
    return [put.path.rsplit("_", 1)[-1] for put in puts]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        pytest.param(503, "SlowDown", id="503"),
        pytest.param(429, "SlowDown", id="429"),
        pytest.param(500, "InternalError", id="500"),
        pytest.param(403, "AccessDenied", id="access-denied-403"),
    ],
)
async def test_retryable_statuses_back_off_three_attempts(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, status: int, code: str | None
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("/test-bucket/", (status, _coded_body(code)))

    await _upload(logger, _payload("req"))

    assert len(bucket.puts()) == 3
    _assert_backoff_gaps(bucket.puts())
    assert len(logger.sink.queued_keys()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        pytest.param(408, None, id="408"),
        pytest.param(502, None, id="502"),
        pytest.param(504, None, id="504"),
    ],
)
async def test_unlisted_statuses_are_not_retried_in_call(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, status: int, code: str | None
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("/test-bucket/", (status, _coded_body(code)))

    started: Final = time.monotonic()
    await _upload(logger, _payload("req"))

    assert len(bucket.puts()) == 1
    assert time.monotonic() - started < 0.5
    assert len(logger.sink.queued_keys()) == 1


@pytest.mark.asyncio
async def test_retry_finishes_before_the_next_first_attempt(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads=1)
    bucket.reply("_a.json", _SLOW_DOWN, (200, ""))
    await _log_all(logger, [_payload("a"), _payload("b")])

    await logger.flush_queue()

    assert _names(bucket.puts()) == ["a.json", "a.json", "b.json"]
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_objects_in_backoff_are_bounded_by_the_slot_width(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads=2)
    bucket.reply("/test-bucket/", _SLOW_DOWN)
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(20)])

    await logger.flush_queue()

    first_and_last: Final[dict[str, list[float]]] = {}
    for put in bucket.puts():
        span = first_and_last.setdefault(put.path, [put.at, put.at])
        span[1] = put.at
    peak: Final = max(
        sum(1 for start, end in first_and_last.values() if start <= moment <= end)
        for moment, _ in first_and_last.values()
    )
    assert peak <= 2, f"{peak} objects slept at once, slot width is 2"
    assert len(bucket.puts()) == 60


@pytest.mark.asyncio
async def test_terminal_code_is_retried_like_base_when_the_drop_flag_is_off(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_drop_on_terminal_error=False)
    bucket.reply("_x.json", (403, "<Error><Code>InvalidRequest</Code></Error>"), (200, ""))

    with _failures() as failures:
        await _upload(logger, _payload("x"))

    assert len(bucket.puts()) == 2
    failures.assert_not_called()

    dropping: Final = _logger(monkeypatch, bucket)
    bucket.script.clear()
    bucket.reply("_x.json", (403, "<Error><Code>InvalidRequest</Code></Error>"), (200, ""))
    before: Final = len(bucket.puts())

    await _upload(dropping, _payload("x"))

    assert len(bucket.puts()) - before == 1


@pytest.mark.asyncio
async def test_send_batch_calls_upload_with_one_positional_arg(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    await _log_all(logger, [_payload("a"), _payload("b")])

    await logger.flush_queue()

    assert sorted(_names(bucket.puts())) == ["a.json", "b.json"]
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_failed_elements_stay_oldest_first_when_requeued(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_retry_age_seconds=3600)
    bucket.reply("_f2.json", _SLOW_DOWN)
    await _log_all(logger, [_payload("f1"), _payload("f2")])

    await asyncio.gather(logger.flush_queue(), _log_once_a_put_arrives(logger, bucket, [_payload("late")]))

    assert logger.sink.queued_keys() == [_key("f2"), _key("late")]


@pytest.mark.asyncio
async def test_overflow_prefers_arrivals_over_failed_elements_without_counting_the_trim(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_queue_size=1)
    bucket.reply("_f2.json", _SLOW_DOWN)
    await _log_all(logger, [_payload("f1"), _payload("f2")])

    with _failures() as mock_failure:
        await asyncio.gather(logger.flush_queue(), _log_once_a_put_arrives(logger, bucket, [_payload("late")]))

    assert logger.sink.queued_keys() == [_key("late")]
    failed_uploads: Final = 1
    assert mock_failure.call_count == failed_uploads


@pytest.mark.asyncio
async def test_fresh_elements_upload_before_stale_retries_after_a_failed_flush(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads=1)
    bucket.reply("_f2.json", _SLOW_DOWN)
    await _log_all(logger, [_payload("f1"), _payload("f2")])

    await asyncio.gather(logger.flush_queue(), _log_once_a_put_arrives(logger, bucket, [_payload("late")]))

    bucket.script.clear()
    before: Final = len(bucket.puts())
    await logger.flush_queue()

    assert _names(bucket.puts()[before:]) == ["late.json", "f2.json"]
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_repeated_overflow_trims_oldest_across_failed_flushes(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_queue_size=3)
    bucket.reply("/test-bucket/", _SLOW_DOWN)
    await _log_all(logger, [_payload(name) for name in ("a", "b", "c")])

    await asyncio.gather(logger.flush_queue(), _log_once_a_put_arrives(logger, bucket, [_payload("d")]))

    assert logger.sink.queued_keys() == [_key("b"), _key("c"), _key("d")]

    seen: Final = len(bucket.puts())

    async def log_e_once_the_next_flush_puts() -> None:
        while len(bucket.puts()) == seen:
            await asyncio.sleep(0.005)
        await log_payload(logger, _payload("e"), START)

    await asyncio.gather(logger.flush_queue(), log_e_once_the_next_flush_puts())

    assert logger.sink.queued_keys() == [_key("c"), _key("d"), _key("e")]


@pytest.mark.asyncio
async def test_retry_age_budget_drops_after_the_clock_set_by_a_partial_failure(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_retry_age_seconds=1)
    bucket.reply("_poison.json", _SLOW_DOWN)
    await _log_all(logger, [_payload("poison"), _payload("good")])

    with _failures() as mock_failure:
        await logger.flush_queue()

        assert logger.sink.queued_keys() == [_key("poison")]

        await asyncio.sleep(1.1)
        await log_payload(logger, _payload("good-2"), START)
        await logger.flush_queue()

    assert logger.sink.queued_keys() == []
    poison_puts: Final = sum(1 for put in bucket.puts() if put.path.endswith("_poison.json"))
    assert poison_puts == 6
    upload_failures: Final = 2
    assert mock_failure.call_count == upload_failures


@pytest.mark.asyncio
async def test_the_retry_clock_starts_at_the_first_partial_failure_not_first_seen(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_retry_age_seconds=1)
    bucket.reply("_poison.json", _SLOW_DOWN)
    await log_payload(logger, _payload("poison"), START)

    await logger.flush_queue()

    assert len(logger.sink.queued_keys()) == 1

    await asyncio.sleep(1.1)
    await log_payload(logger, _payload("good"), START)
    await logger.flush_queue()

    assert logger.sink.queued_keys() == [_key("poison")]


@pytest.mark.asyncio
async def test_an_unlisted_5xx_is_requeued_without_an_extra_attempt(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("/test-bucket/", (507, ""))

    await _upload(logger, _payload("req-507"))

    assert len(logger.sink.queued_keys()) == 1
    assert len(bucket.puts()) == 1


async def _peak_width(logger: S3Logger, bucket: ScriptedBucket, events: int) -> int:
    bucket.peak = 0
    bucket.delay = 0.1
    await _log_all(logger, [_payload(f"width-{index}") for index in range(events)])
    await logger.flush_queue()
    bucket.delay = 0.0
    return bucket.peak


@pytest.mark.asyncio
async def test_configured_concurrency_is_the_fixed_limit_when_adaptive_is_off(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _logger(monkeypatch, bucket, s3_max_concurrent_uploads=64)

    assert await _peak_width(logger, bucket, 128) == 64

    bucket.reply("_elem-0.json", _SLOW_DOWN, (200, ""))
    await _upload(logger, _payload("elem-0"))

    assert await _peak_width(logger, bucket, 128) == 64


@pytest.mark.asyncio
async def test_default_upload_width_is_16(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    from litellm.constants import DEFAULT_S3_MAX_CONCURRENT_UPLOADS

    logger: Final = _logger(monkeypatch, bucket)

    assert await _peak_width(logger, bucket, 2 * DEFAULT_S3_MAX_CONCURRENT_UPLOADS) == DEFAULT_S3_MAX_CONCURRENT_UPLOADS
    assert DEFAULT_S3_MAX_CONCURRENT_UPLOADS == 16


async def _timed_send_batch(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, size: int) -> float:
    logger: Final = _logger(monkeypatch, bucket)
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(size)])
    started: Final = time.perf_counter()
    await logger.flush_queue()
    return time.perf_counter() - started


@pytest.mark.asyncio
async def test_send_batch_time_grows_linearly_with_the_batch(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    baseline: Final = await _timed_send_batch(monkeypatch, bucket, 2_000)
    quadrupled: Final = await _timed_send_batch(monkeypatch, bucket, 8_000)

    assert quadrupled / baseline < 8, f"2k took {baseline:.3f}s, 8k took {quadrupled:.3f}s"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.asyncio
async def test_repeated_failure_notifications_upload_once(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    stream: bool,
    concurrent: bool,
) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    sink: Final = _logger(monkeypatch, bucket)
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [sink])
    logging_obj: Final = Logging(
        model="openai/gpt-4o-mini",
        messages=[],
        stream=stream,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="synthetic-request",
        function_id="synthetic-function",
    )
    error: Final = litellm.ServiceUnavailableError(
        message="synthetic deployment-selection failure",
        llm_provider="openai",
        model="openai/gpt-4o-mini",
    )

    if concurrent:
        await asyncio.gather(*(logging_obj.async_failure_handler(error, "synthetic traceback") for _ in range(3)))
    else:
        for _ in range(3):
            await logging_obj.async_failure_handler(error, "synthetic traceback")

    assert len(sink.sink.queued_keys()) == 1
    await sink.flush_queue()
    await asyncio.sleep(0)
    assert len(bucket.puts()) == 1
    assert all(_uploaded(put)["status"] == "failure" for put in bucket.puts())


@pytest.fixture
def _s3_v2_callback_params(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])


async def _wait_for_a_queued_key(logger: S3Logger) -> list[str]:
    for _ in range(500):
        keys = logger.sink.queued_keys()
        if keys:
            return keys
        await asyncio.sleep(0.01)
    return []


@pytest.mark.usefixtures("_s3_v2_callback_params")
@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True])
async def test_basic_s3_v2_logging(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, streaming: bool) -> None:
    monkeypatch.setattr(
        litellm,
        "s3_callback_params",
        {
            "s3_bucket_name": "load-testing-oct",
            "s3_endpoint_url": bucket.url,
            "s3_aws_secret_access_key": "test-secret",
            "s3_aws_access_key_id": "test-key",
            "s3_region_name": "us-west-2",
        },
    )

    s3_v2_logger: Final = _logger_from_globals(monkeypatch)
    monkeypatch.setattr(litellm, "callbacks", [s3_v2_logger])

    response_id = None
    response = await litellm.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "This is a test"}],
        mock_response="It's simple to use and easy to get started",
        stream=streaming,
    )
    if streaming:
        async for chunk in response:
            response_id = chunk.id
    else:
        response_id = response.id

    assert await _wait_for_a_queued_key(s3_v2_logger)
    await s3_v2_logger.flush_queue()

    uploaded_keys: Final = [unquote(put.path) for put in bucket.puts()]
    assert len(uploaded_keys) > 0, "S3 upload was never called"
    assert response_id is not None
    assert any(response_id in key for key in uploaded_keys), (
        f"Expected response_id={response_id} in one of the uploaded S3 keys: {uploaded_keys}"
    )


def _logger_from_globals(monkeypatch: pytest.MonkeyPatch) -> S3Logger:
    logger: Final = rust_s3_logger(monkeypatch)
    _ = logger.periodic_flush_task.cancel()
    return logger


@pytest.mark.usefixtures("_s3_v2_callback_params")
@pytest.mark.asyncio
async def test_basic_s3_v2_logging_failure(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket) -> None:
    monkeypatch.setattr(
        litellm,
        "s3_callback_params",
        {
            "s3_bucket_name": "test-bucket",
            "s3_endpoint_url": bucket.url,
            "s3_aws_secret_access_key": "test-secret",
            "s3_aws_access_key_id": "test-key",
            "s3_region_name": "us-west-2",
        },
    )
    s3_v2_logger: Final = _logger_from_globals(monkeypatch)
    monkeypatch.setattr(litellm, "callbacks", [s3_v2_logger])

    with contextlib.suppress(Exception):
        await litellm.acompletion(
            model="gpt-5-mini",
            api_key="invalid-api-key",
            messages=[{"role": "user", "content": "This is a test"}],
            mock_response=Exception("forced failure for S3 logging test"),
        )

    assert await _wait_for_a_queued_key(s3_v2_logger)
    await s3_v2_logger.flush_queue()

    (put,) = bucket.puts()
    assert put.path.startswith("/test-bucket/")
    assert put.headers["content-type"] == "application/json"
    assert '"model": "gpt-5-mini"' in put.body.decode()


def _batch_line_items_payload() -> dict[str, object]:
    line = {
        "batch_id": "bGl0ZWxsbV9wcm94eTtiYXRjaA",
        "custom_id": "ok-1",
        "status": "success",
        "request": {"custom_id": "ok-1", "body": {"messages": [{"role": "user", "content": "hi"}]}},
        "response": {"custom_id": "ok-1", "response": {"status_code": 200, "body": {"id": "chatcmpl-1"}}},
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "response_cost": 0.005,
        "model": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "metadata": {"user_api_key_hash": "hash-alice"},
    }
    return {
        "batch_id": "bGl0ZWxsbV9wcm94eTtiYXRjaA",
        "finished_at": 1791494130,
        "part": 0,
        "lines": (line, {**line, "custom_id": "ok-2"}),
    }


def _s3_logger_answering(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, status: int, s3_path: str | None, content: bytes = b""
) -> S3Logger:
    bucket.reply("/test-bucket/", (status, content.decode()))
    return _logger(
        monkeypatch,
        bucket,
        s3_path=s3_path,
        s3_aws_access_key_id=_ACCESS_KEY,
        s3_aws_secret_access_key=_SECRET_KEY,
        s3_flush_interval=3600,
    )


@pytest.mark.asyncio
async def test_batch_line_items_are_one_ndjson_object_under_the_configured_path(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _s3_logger_answering(monkeypatch, bucket, 200, "cold/")
    payload: Final = _batch_line_items_payload()

    await logger.async_log_batch_line_items(payload)

    (request,) = bucket.received
    assert request.method == "PUT"
    assert request.path == "/test-bucket/cold/batch_line_items/2026-10-08/bGl0ZWxsbV9wcm94eTtiYXRjaA/part-00000.jsonl"
    assert request.headers["content-type"] == "application/x-ndjson"
    assert request.headers["content-disposition"] == 'inline; filename="bGl0ZWxsbV9wcm94eTtiYXRjaA-part-00000.jsonl"'
    assert [json.loads(line) for line in request.body.decode().splitlines()] == list(payload["lines"])
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_batch_line_items_without_an_s3_path_sit_at_the_bucket_root(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _s3_logger_answering(monkeypatch, bucket, 200, None)

    await logger.async_log_batch_line_items(_batch_line_items_payload())

    assert [request.path for request in bucket.received] == [
        "/test-bucket/batch_line_items/2026-10-08/bGl0ZWxsbV9wcm94eTtiYXRjaA/part-00000.jsonl"
    ]


@pytest.mark.asyncio
async def test_a_rejected_batch_line_items_upload_is_logged_with_the_batch_id(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    logger: Final = _s3_logger_answering(monkeypatch, bucket, 400, "cold")

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        await logger.async_log_batch_line_items(_batch_line_items_payload())

    assert len(bucket.received) == 1
    assert any(
        "bGl0ZWxsbV9wcm94eTtiYXRjaA" in record.getMessage() and "were not stored" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_batch_line_items_object_the_bucket_refuses_for_good_is_not_kept_in_memory(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _s3_logger_answering(
        monkeypatch, bucket, 400, "cold", b"<Error><Code>InvalidArgument</Code></Error>"
    )

    await logger.async_log_batch_line_items(_batch_line_items_payload())
    await logger.flush_queue()

    assert len(bucket.received) == 1
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_batch_line_items_are_serialized_and_hashed_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    import threading

    logger: Final = _s3_logger_answering(monkeypatch, bucket, 200, "cold/")
    serialize: Final = s3_v3.batch_line_items_ndjson
    threads: Final[dict[str, threading.Thread]] = {}

    def _serialize(payload: object) -> str:
        threads["serialize"] = threading.current_thread()
        return serialize(payload)

    monkeypatch.setattr(s3_v3, "batch_line_items_ndjson", _serialize)

    await logger.async_log_batch_line_items(_batch_line_items_payload())

    assert len(bucket.received) == 1
    assert set(threads) == {"serialize"}
    assert threading.main_thread() not in threads.values()


_BOUNDING_START: Final = datetime(2026, 8, 24, 6, 18, 41, 948021)


def _oversized_response_id() -> str:
    return "resp_" + "A" * 1100


def _alias_metadata(team_alias: str | None = None, key_alias: str | None = None) -> dict[str, object]:
    metadata: Final[dict[str, object]] = {}
    if team_alias is not None:
        metadata["user_api_key_team_alias"] = team_alias
    if key_alias is not None:
        metadata["user_api_key_alias"] = key_alias
    return metadata


async def _bounded_key(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    response_id: str,
    s3_path: str = "",
    team_alias: str | None = None,
    key_alias: str | None = None,
    start_time: datetime = _BOUNDING_START,
) -> str:
    logger: Final = _logger(
        monkeypatch,
        bucket,
        s3_path=s3_path or None,
        s3_use_team_prefix=team_alias is not None,
        s3_use_key_prefix=key_alias is not None,
    )
    return await _queued_key(logger, _payload(response_id, metadata=_alias_metadata(team_alias, key_alias)), start_time)


@pytest.mark.asyncio
async def test_s3_object_key_bounds_the_file_segment_when_the_full_key_fits(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    file_name: Final = "time-01-14-19-015434_resp_" + "A" * 300

    key: Final = await _bounded_key(
        monkeypatch, bucket, "resp_" + "A" * 300, "requests", start_time=datetime(2026, 9, 12, 1, 14, 19, 15434)
    )

    file_segment: Final = key.rsplit("/", 1)[1]
    assert len(key.encode("utf-8")) < MAX_S3_OBJECT_KEY_BYTES
    assert len(file_segment.encode("utf-8")) <= 255
    assert file_segment.startswith("time-01-14-19-015434_resp_")
    assert file_segment.endswith(f"_{hashlib.sha256(file_name.encode('utf-8')).hexdigest()}.json")


@pytest.mark.asyncio
async def test_s3_object_key_file_segment_limit_counts_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    key: Final = await _bounded_key(
        monkeypatch, bucket, "resp_" + "日" * 100, "requests", start_time=datetime(2026, 9, 12, 1, 14, 19, 15434)
    )

    file_segment: Final = key.rsplit("/", 1)[1]
    assert len(file_segment.encode("utf-8")) <= 255
    assert "�" not in file_segment


@pytest.mark.asyncio
async def test_s3_object_key_is_bounded_for_oversized_response_id(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    file_name: Final = f"time-06-18-41-948021_{_oversized_response_id()}"

    key: Final = await _bounded_key(
        monkeypatch, bucket, _oversized_response_id(), "input", team_alias="DefaultTeamProd"
    )

    assert len(key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert key.startswith("input/DefaultTeamProd/2026-08-24/time-06-18-41-948021_resp_")
    assert key.endswith(f"_{hashlib.sha256(file_name.encode('utf-8')).hexdigest()}.json")


@pytest.mark.parametrize(
    "s3_path,team_alias,key_alias",
    [
        ("input", None, None),
        ("a" * 900, None, None),
        ("input", "team-" + "b" * 900, None),
        ("c" * 600, "team-" + "d" * 600, "key-" + "e" * 600),
        ("", "ssss/" * 199 + "ssss", None),
    ],
)
@pytest.mark.asyncio
async def test_s3_object_key_is_bounded_for_long_paths_and_aliases(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    s3_path: str,
    team_alias: str | None,
    key_alias: str | None,
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    key: Final = await _bounded_key(monkeypatch, bucket, _oversized_response_id(), s3_path, team_alias, key_alias)

    assert len(key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert key.endswith(".json")
    assert "/2026-08-24/" in key or key.startswith("2026-08-24/")
    assert "/" not in key.rsplit("2026-08-24/", 1)[1]


@pytest.mark.asyncio
async def test_s3_object_key_trimmed_prefixes_stay_distinct_per_operator(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    keys: Final = [
        await _bounded_key(monkeypatch, bucket, _oversized_response_id(), "input", "team-" + "b" * 1000 + suffix)
        for suffix in ("-one", "-two")
    ]

    assert keys[0] != keys[1]
    assert all(key.startswith("input/team-" + "b" * 900) for key in keys)
    assert all(len(key.encode("utf-8")) == MAX_S3_OBJECT_KEY_BYTES for key in keys)


@pytest.mark.asyncio
async def test_s3_object_key_bounded_prefix_never_splits_a_multibyte_character(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    s3_path: Final = "日本語" * 200

    key: Final = await _bounded_key(monkeypatch, bucket, _oversized_response_id(), s3_path, "チーム" * 200)

    assert len(key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert key.startswith(s3_path[:100])
    assert "�" not in key


@pytest.mark.asyncio
async def test_s3_object_key_stays_unique_for_ids_sharing_a_head(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    keys: Final = {
        await _bounded_key(monkeypatch, bucket, f"{_oversized_response_id()}{suffix}", "input")
        for suffix in ("first", "second", "third")
    }

    assert len(keys) == 3


@pytest.mark.asyncio
async def test_s3_object_key_bounding_matches_the_documented_layout(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    file_name: Final = f"time-06-18-41-948021_{_oversized_response_id()}"

    key: Final = await _bounded_key(monkeypatch, bucket, _oversized_response_id(), "input", "team")

    digest: Final = hashlib.sha256(file_name.encode("utf-8")).hexdigest()
    assert key == f"input/team/2026-08-24/{file_name[:64]}_{digest}.json"


@pytest.mark.asyncio
async def test_s3_object_key_keeps_the_configured_prefix_when_only_the_id_overflows(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    prefix: Final = "team-" + "b" * 934 + "/"

    key: Final = await _bounded_key(monkeypatch, bucket, _oversized_response_id(), "", prefix[:-1])

    assert key.startswith(prefix + "2026-08-24/")
    assert len(key.encode("utf-8")) == MAX_S3_OBJECT_KEY_BYTES


@pytest.mark.asyncio
async def test_s3_object_key_spends_the_whole_budget_when_the_prefix_must_be_trimmed(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    key: Final = await _bounded_key(monkeypatch, bucket, "abc", "p" * 400 + "/" + "q" * 600)

    assert len(key.encode("utf-8")) == MAX_S3_OBJECT_KEY_BYTES
    assert key.startswith("p" * 400 + "/" + "q" * 500)


@pytest.mark.asyncio
async def test_s3_object_key_keeps_a_single_segment_path_as_far_as_it_fits(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    key: Final = await _bounded_key(monkeypatch, bucket, "chatcmpl-xyz", "a" * 1050)

    assert len(key.encode("utf-8")) == MAX_S3_OBJECT_KEY_BYTES
    assert key.startswith("a" * 900)


@pytest.mark.asyncio
async def test_create_s3_batch_logging_element_bounds_key_and_keeps_full_response_id(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    logger: Final = _logger(monkeypatch, bucket, s3_use_team_prefix=True, s3_use_key_prefix=True)
    response_id: Final = _oversized_response_id()

    await _upload(
        logger, _payload(response_id, metadata=_alias_metadata("DefaultTeamProd", "prod-key")), _BOUNDING_START
    )

    (put,) = bucket.puts()
    s3_object_key: Final = unquote(put.path).removeprefix("/test-bucket/")
    assert len(s3_object_key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert s3_object_key.startswith("DefaultTeamProd/prod-key/2026-08-24/")
    assert _uploaded(put)["id"] == response_id


async def _download_filenames(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, response_ids: list[str]
) -> list[str]:
    logger: Final = _logger(monkeypatch, bucket)
    await _log_all_at(logger, [_payload(response_id) for response_id in response_ids], _BOUNDING_START)
    await logger.flush_queue()
    names: Final = [
        re.fullmatch(r'inline; filename="(.*)"', put.headers["content-disposition"]) for put in bucket.puts()
    ]
    return [name.group(1) for name in names if name is not None]


async def _log_all_at(logger: S3Logger, payloads: list[dict[str, object]], start_time: datetime) -> None:
    for payload in payloads:
        await log_payload(logger, payload, start_time)


@pytest.mark.asyncio
async def test_s3_object_download_filename_is_bounded_for_oversized_response_id(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES

    (file_name,) = await _download_filenames(monkeypatch, bucket, [_oversized_response_id()])

    assert len(file_name.encode("utf-8")) <= MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES
    assert file_name.startswith("time-2026-08-24T06-18-41-948021_resp_")
    assert file_name.endswith(".json")


@pytest.mark.asyncio
async def test_s3_object_download_filenames_stay_distinct_when_shortened(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    file_names: Final = set(
        await _download_filenames(
            monkeypatch, bucket, [_oversized_response_id() + suffix for suffix in ("first", "second", "third")]
        )
    )

    assert len(file_names) == 3


@pytest.mark.asyncio
async def test_s3_object_download_filename_short_id_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    (file_name,) = await _download_filenames(monkeypatch, bucket, ["resp_abc123"])

    assert file_name == "time-2026-08-24T06-18-41-948021_resp_abc123.json"


@pytest.mark.asyncio
async def test_create_s3_batch_logging_element_bounds_the_download_filename(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES

    (file_name,) = await _download_filenames(monkeypatch, bucket, [_oversized_response_id()])

    assert len(file_name.encode("utf-8")) <= MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES


@pytest.mark.asyncio
async def test_audit_log_object_key_is_bounded_for_a_long_configured_path(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.constants import MAX_S3_OBJECT_KEY_BYTES

    logger: Final = _logger(monkeypatch, bucket, s3_path="audit-archive/" + "z" * 1100)

    await logger.async_log_audit_log_event({"id": "1a4f7bd0-6f1e-4d0a-9b3c-9f2e1d5a7c88"})

    (key,) = logger.sink.queued_keys()
    assert len(key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES
    assert key.startswith("audit-archive/" + "z" * 900)


@pytest.mark.asyncio
async def test_s3_object_download_filename_drops_characters_that_break_the_header(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    (file_name,) = await _download_filenames(monkeypatch, bucket, ['resp_a"b/c'])

    assert file_name == "time-2026-08-24T06-18-41-948021_resp_a_b_c.json"


@pytest.mark.asyncio
async def test_s3_callback_params_override_uses_alternate_dict(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setattr(litellm, "s3_callback_params", {"s3_bucket_name": "normal-bucket"})
    logger: Final = rust_s3_logger(
        monkeypatch,
        s3_bucket_name="audit-bucket",
        s3_path="audit-prefix",
        s3_region_name="us-west-2",
        s3_endpoint_url=bucket.url,
        s3_aws_access_key_id="test-key",
        s3_aws_secret_access_key="test-secret",
    )
    _ = logger.periodic_flush_task.cancel()

    await _upload(logger, _payload("override"))

    (put,) = bucket.puts()
    assert put.path.startswith("/audit-bucket/")
    assert put.path.startswith("/audit-bucket/audit-prefix/")
    assert "/us-west-2/s3/aws4_request" in put.headers["authorization"]


@pytest.mark.asyncio
async def test_s3_callback_params_override_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setenv("MY_AUDIT_BUCKET", "resolved-bucket")
    override: Final = {
        "s3_bucket_name": "os.environ/MY_AUDIT_BUCKET",
        "s3_endpoint_url": bucket.url,
        "s3_aws_access_key_id": "test-key",
        "s3_aws_secret_access_key": "test-secret",
        "s3_region_name": "us-east-1",
    }
    monkeypatch.setattr(litellm, "s3_callback_params", {"s3_bucket_name": "os.environ/MY_AUDIT_BUCKET"})
    logger: Final = rust_s3_logger(monkeypatch, **override)
    _ = logger.periodic_flush_task.cancel()

    await _upload(logger, _payload("override"))

    assert bucket.puts()[0].path.startswith("/resolved-bucket/")
    assert override["s3_bucket_name"] == "os.environ/MY_AUDIT_BUCKET"
    assert litellm.s3_callback_params["s3_bucket_name"] == "os.environ/MY_AUDIT_BUCKET"


@pytest.mark.asyncio
async def test_s3_callback_params_override_none_falls_back_to_global(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.setattr(
        litellm,
        "s3_callback_params",
        {
            "s3_bucket_name": "from-global",
            "s3_endpoint_url": bucket.url,
            "s3_aws_access_key_id": "test-key",
            "s3_aws_secret_access_key": "test-secret",
            "s3_region_name": "us-east-1",
        },
    )
    logger: Final = _logger_from_globals(monkeypatch)

    await _upload(logger, _payload("global"))

    assert bucket.puts()[0].path.startswith("/from-global/")


_SIGV4_ACCESS_KEY: Final = re.compile(r"Credential=(AKIA\d+)/")


@pytest.fixture
def rotating_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    import sys
    import textwrap
    import uuid

    counter: Final = tmp_path / "generation"
    script: Final = tmp_path / "rotate_credentials.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, sys
            from datetime import datetime, timedelta, timezone
            from pathlib import Path

            counter = Path({str(counter)!r})
            generation = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(generation + 1))
            expiry = (datetime.now(timezone.utc) + timedelta(minutes=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
            json.dump(
                {{
                    "Version": 1,
                    "AccessKeyId": f"AKIA{{generation}}",
                    "SecretAccessKey": f"secret-{{generation}}",
                    "SessionToken": f"token-{{generation}}",
                    "Expiration": expiry,
                }},
                sys.stdout,
            )
            """
        )
    )
    profile: Final = f"rotating-{uuid.uuid4().hex}"
    (tmp_path / "config").write_text(f"[profile {profile}]\ncredential_process = {sys.executable} {script}\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    return profile


def _generation(request: Received) -> tuple[str, str]:
    access_key: Final = _SIGV4_ACCESS_KEY.search(request.headers["authorization"])
    assert access_key is not None
    return access_key.group(1).removeprefix("AKIA"), request.headers["x-amz-security-token"].removeprefix("token-")


def _profile_logger(monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, profile: str) -> S3Logger:
    logger: Final = rust_s3_logger(
        monkeypatch,
        s3_bucket_name="test-bucket",
        s3_endpoint_url=bucket.url,
        s3_region_name="us-east-1",
        s3_aws_profile_name=profile,
        s3_flush_interval=3600,
    )
    _ = logger.periodic_flush_task.cancel()
    return logger


_SIGNATURE_DOES_NOT_MATCH: Final = "<Error><Code>SignatureDoesNotMatch</Code></Error>"


@pytest.mark.asyncio
async def test_async_upload_signs_with_one_frozen_credential_snapshot(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, rotating_profile: str
) -> None:
    await _upload(_profile_logger(monkeypatch, bucket, rotating_profile), _payload("frozen"))

    assert len(bucket.puts()) == 1
    access_key_generation, token_generation = _generation(bucket.puts()[0])
    assert access_key_generation == token_generation


@pytest.mark.asyncio
async def test_async_upload_retries_403_with_fresh_credentials_and_signature(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, rotating_profile: str
) -> None:
    bucket.reply("_test-403.json", (403, _SIGNATURE_DOES_NOT_MATCH), (200, ""))
    logger: Final = _profile_logger(monkeypatch, bucket, rotating_profile)

    await _upload(logger, _payload("test-403"))

    requests: Final = bucket.puts()
    assert len(requests) == 2
    first_key, first_token = _generation(requests[0])
    second_key, second_token = _generation(requests[1])
    assert first_key == first_token
    assert second_key == second_token
    assert int(second_key) > int(first_key)
    assert requests[1].headers["authorization"] != requests[0].headers["authorization"]
    _assert_backoff_gaps(requests)
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_async_upload_exhausts_403_retries_through_production_http_handler(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, rotating_profile: str
) -> None:
    bucket.reply("_test-403-exhausted.json", (403, _SIGNATURE_DOES_NOT_MATCH))
    logger: Final = _profile_logger(monkeypatch, bucket, rotating_profile)

    with _failures() as failure:
        await _upload(logger, _payload("test-403-exhausted"))

    assert len(bucket.puts()) == 3
    _assert_backoff_gaps(bucket.puts())
    failure.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("transient_status", [500, 503])
async def test_async_upload_recovers_from_transient_5xx_through_production_http_handler(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, rotating_profile: str, transient_status: int
) -> None:
    bucket.reply(f"_test-{transient_status}.json", (transient_status, _SIGNATURE_DOES_NOT_MATCH), (200, ""))
    logger: Final = _profile_logger(monkeypatch, bucket, rotating_profile)

    with _failures() as failure:
        await _upload(logger, _payload(f"test-{transient_status}"))

    requests: Final = bucket.puts()
    assert len(requests) == 2
    assert all(request.method == "PUT" for request in requests)
    assert requests[0].path == requests[1].path
    assert requests[0].body == requests[1].body
    _assert_backoff_gaps(requests)
    failure.assert_not_called()
    assert logger.sink.queued_keys() == []


_CONSTRUCTOR_BASE: Final[S3LoggerSettings] = {
    "s3_bucket_name": "test-bucket",
    "s3_aws_access_key_id": "test-key",
    "s3_aws_secret_access_key": "test-secret",
    "s3_region_name": "us-east-1",
}


def _constructed(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    settings: S3LoggerSettings,
    override: Mapping[str, object] | None = None,
    *,
    configured: Mapping[str, object] | None = None,
    periodic: bool = False,
) -> S3Logger:
    monkeypatch.setattr(litellm, "s3_callback_params", configured)
    logger: Final = S3Logger(override, **(_CONSTRUCTOR_BASE | {"s3_endpoint_url": bucket.url} | settings))
    if not periodic:
        _ = logger.periodic_flush_task.cancel()
    return logger


@pytest.mark.asyncio
async def test_constructor_keywords_build_a_logger_that_uploads_under_them(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_path": "from-constructor"})

    await _upload(logger, _payload("ctor"))

    assert isinstance(logger, S3Logger)
    assert logger.s3_path == "from-constructor"
    assert [put.path for put in bucket.puts()] == [f"/test-bucket/from-constructor/{_key('ctor')}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [
        pytest.param(None, {"s3_path": "configured"}, "configured", id="override-wins"),
        pytest.param({"s3_path": "global"}, None, "global", id="global-wins"),
        pytest.param({"s3_path": "global"}, {}, "constructor", id="empty-override-is-opt-in"),
        pytest.param(None, {"s3_path": ""}, "constructor", id="blank-falls-back"),
    ],
)
async def test_callback_params_text_settings_win_over_constructor(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    configured: Mapping[str, object] | None,
    override: Mapping[str, object] | None,
    expected: str,
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_path": "constructor"}, override, configured=configured)

    await _upload(logger, _payload("text"))

    assert [put.path for put in bucket.puts()] == [f"/test-bucket/{expected}/{_key('text')}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constructor", "configured", "expected"),
    [
        pytest.param("hour", "day", "hour", id="constructor-wins"),
        pytest.param(None, "hour", "hour", id="configured-when-constructor-unset"),
    ],
)
async def test_constructor_partition_granularity_wins_over_callback_params(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, constructor: str | None, configured: str, expected: str
) -> None:
    logger: Final = _constructed(
        monkeypatch, bucket, {"s3_partition_granularity": constructor}, {"s3_partition_granularity": configured}
    )

    assert logger.resolve_partition_granularity() == expected


@pytest.mark.asyncio
async def test_constructor_prompts_only_wins_over_callback_params(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)
    logger: Final = _constructed(monkeypatch, bucket, {"s3_log_prompts_only": True}, {"s3_log_prompts_only": False})

    await _upload(logger, _payload("prompts", response={"choices": [{"message": {"content": "secret answer"}}]}))

    (put,) = bucket.puts()
    assert b"secret answer" not in put.body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constructor", "override", "kept"),
    [
        pytest.param(7, {"s3_max_queue_size": 4}, 4, id="callback-params-win"),
        pytest.param(4, {"s3_max_queue_size": "abc"}, 4, id="invalid-callback-params-fall-back-to-constructor"),
        pytest.param(4, {}, 4, id="constructor-only"),
    ],
)
async def test_queue_size_resolves_callback_params_then_constructor(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    constructor: int,
    override: Mapping[str, object],
    kept: int,
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_max_queue_size": constructor}, override)
    bucket.reply("/test-bucket/", _SLOW_DOWN)
    await _log_all(logger, [_payload("first"), _payload("second")])

    with _failures():
        await asyncio.gather(
            logger.flush_queue(),
            _log_once_a_put_arrives(logger, bucket, [_payload(f"late-{index}") for index in range(3)]),
        )

    assert len(logger.sink.queued_keys()) == kept


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constructor", "override", "still_queued"),
    [
        pytest.param(0, {}, True, id="constructor-zero-disables-the-budget"),
        pytest.param(None, {}, True, id="constructor-none-disables-the-budget"),
        pytest.param(1, {"s3_max_retry_age_seconds": "abc"}, False, id="invalid-callback-params-fall-back"),
        pytest.param(3600, {"s3_max_retry_age_seconds": 1}, False, id="callback-params-win"),
        pytest.param(1, {"s3_max_retry_age_seconds": ""}, False, id="blank-callback-params-fall-back"),
    ],
)
async def test_retry_age_resolves_callback_params_then_constructor(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    constructor: int | None,
    override: Mapping[str, object],
    still_queued: bool,
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_max_retry_age_seconds": constructor}, override)
    bucket.reply("_poison.json", _SLOW_DOWN)
    await _log_all(logger, [_payload("poison"), _payload("good")])

    with _failures():
        await logger.flush_queue()
        await asyncio.sleep(1.1)
        await log_payload(logger, _payload("good-2"), START)
        await logger.flush_queue()

    assert logger.sink.queued_keys() == ([_key("poison")] if still_queued else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constructor", "override", "dropped"),
    [
        pytest.param(False, {"s3_drop_on_terminal_error": True}, True, id="callback-params-win"),
        pytest.param(False, {}, False, id="constructor-opts-out"),
        pytest.param(False, {"s3_drop_on_terminal_error": "banana"}, True, id="invalid-callback-params-drop"),
    ],
)
async def test_drop_on_terminal_error_resolves_callback_params_then_constructor(
    monkeypatch: pytest.MonkeyPatch,
    bucket: ScriptedBucket,
    constructor: bool,
    override: Mapping[str, object],
    dropped: bool,
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_drop_on_terminal_error": constructor}, override)
    bucket.reply("_elem-1.json", (400, "<Error><Code>EntityTooLarge</Code></Error>"))
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(5)])

    with _failures():
        await logger.flush_queue()

    assert logger.sink.queued_keys() == ([] if dropped else [_key("elem-1")])


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, ""])
async def test_empty_callback_params_concurrency_falls_back_to_constructor_value(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, empty: object
) -> None:
    logger: Final = _constructed(
        monkeypatch, bucket, {"s3_max_concurrent_uploads": 4}, {"s3_max_concurrent_uploads": empty}
    )

    assert await _peak_width(logger, bucket, 12) == 4


@pytest.mark.asyncio
async def test_callback_params_concurrency_wins_over_constructor(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _constructed(
        monkeypatch, bucket, {"s3_max_concurrent_uploads": 4}, {"s3_max_concurrent_uploads": 2}
    )

    assert await _peak_width(logger, bucket, 12) == 2


@pytest.mark.asyncio
async def test_constructor_batch_size_flushes_audit_logs_once_the_queue_reaches_it(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_batch_size": 2})

    await logger.async_log_audit_log_event({"id": "audit-1"})

    assert len(logger.sink.queued_keys()) == 1
    assert bucket.puts() == []

    await logger.async_log_audit_log_event({"id": "audit-2"})

    assert logger.sink.queued_keys() == []
    assert len(bucket.puts()) == 2


@pytest.mark.asyncio
async def test_constructor_flush_interval_drives_the_periodic_flush(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    logger: Final = _constructed(monkeypatch, bucket, {"s3_flush_interval": 1}, periodic=True)
    try:
        await log_payload(logger, _payload("tick"), START)
        for _ in range(100):
            if bucket.puts():
                break
            await asyncio.sleep(0.05)

        assert [put.path for put in bucket.puts()] == [f"/test-bucket/{_key('tick')}"]
    finally:
        _ = logger.periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_the_declared_subscription_matches_the_rust_manifest(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.integrations.custom_logger_v2.rust import WireManifest

    logger: Final = _logger(monkeypatch, bucket)
    manifest: Final = WireManifest.model_validate_json(logger.rust.manifest())

    assert {(s.hook, s.reads, s.writes, s.may_abort, s.parallel, s.awaited) for s in manifest.subscriptions} == {
        (hook, s.reads, s.writes, s.may_abort, s.parallel, s.awaited) for hook, s in S3Logger.subscriptions.items()
    }
    assert logger.selection is not None
    assert manifest.selection.name == logger.selection.name
    assert frozenset(manifest.selection.aliases) == logger.selection.aliases
    assert manifest.selection.activation == logger.selection.activation
    assert frozenset(manifest.selection.opt_out_by) == logger.selection.opt_out_by
    assert manifest.selection.honors_no_log == logger.selection.honors_no_log
    assert not manifest.has_request_state


@pytest.mark.asyncio
async def test_the_s3_v2_callback_name_resolves_to_the_rust_logger_class(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    logger: Final = _logger(monkeypatch, bucket)

    assert CustomLoggerRegistry.get_class_type_for_custom_logger_name("s3_v2") is S3Logger
    assert CustomLoggerRegistry.get_callback_str_from_class_type(type(logger)) == "s3_v2"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: Final[list[logging.LogRecord]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_a_partially_failed_flush_logs_s3_batch_upload_error_like_main(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    from litellm._logging import verbose_logger
    from litellm.integrations.s3_v2 import S3BatchUploadError

    logger: Final = _logger(monkeypatch, bucket)
    bucket.reply("_elem-1.json", _SLOW_DOWN)
    await _log_all(logger, [_payload(f"elem-{i}") for i in range(3)])
    records: Final = _Records()
    verbose_logger.addHandler(records)
    try:
        await logger.flush_queue()
    finally:
        verbose_logger.removeHandler(records)

    failures: Final = [record for record in records.records if record.exc_info is not None]
    assert len(failures) == 1
    record: Final = failures[0]
    assert record.name == "LiteLLM"
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "CustomLogger: async_send_batch raised; preserving 3 events in queue for retry"
    assert record.exc_info is not None
    error: Final = record.exc_info[1]
    assert isinstance(error, S3BatchUploadError)
    assert (error.failed, error.total) == (1, 3)
    assert str(error) == "1 of 3 S3 uploads failed; transient failures kept in queue for the next flush"
    assert "S3BatchUploadError: 1 of 3 S3 uploads failed" in logging.Formatter().formatException(record.exc_info)
