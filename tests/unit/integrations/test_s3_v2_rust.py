import base64
import hashlib
import json
from collections.abc import Iterator, Mapping
from datetime import datetime
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
from litellm.integrations.s3_v3 import S3LoggerV3
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.rust_bridge import _native

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
    s3_v3.native_s3_v2_enabled.cache_clear()


@pytest.fixture
def bucket() -> Iterator[ScriptedBucket]:
    with scripted_bucket() as scripted:
        yield scripted


def _logger(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, **params: object
) -> S3LoggerV3:  # kwargs-ok: S3Logger keywords
    return rust_s3_logger(
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


def _payload(response_id: str = "test", **fields: object) -> dict[str, object]:  # kwargs-ok: payload fields
    return {"id": response_id, "metadata": {}, "messages": [], **fields}


async def _upload(logger: S3LoggerV3, payload: Mapping[str, object], start_time: datetime = START) -> None:
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


async def _queued_key(logger: S3LoggerV3, payload: Mapping[str, object], start_time: datetime = START) -> str:
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
) -> S3LoggerV3:  # kwargs-ok: S3Logger keywords
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
