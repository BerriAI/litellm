import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import reduce
from typing import Final
from uuid import UUID

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Accepted, Batch, Delivered, Rejected, Undelivered, build_pipeline
from litellm.proxy.control_plane.audit.s3 import HttpxObjectStore, S3ObjectLockSink, object_key
from litellm.proxy.control_plane.settings import S3SinkSettings

T0: Final = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def test_empty_s3_prefix_does_not_add_a_leading_slash() -> None:
    assert not object_key("", "run", _batch(1)).startswith("/")


SETTINGS: Final = S3SinkSettings(
    bucket="audit",
    prefix="cp",
    region="eu-west-1",
    endpoint_url=None,
    object_lock_mode="COMPLIANCE",
    retention_days=30,
    timeout_seconds=5.0,
)


@dataclass
class FakeStore:  # mutable-ok: scripts responses and records puts for assertions
    outcomes: list[httpx.Response | Exception]
    puts: list[tuple[str, bytes, Mapping[str, str], float]] = field(default_factory=list)

    async def put(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.puts.append((url, content, headers, timeout))
        outcome: Final = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_s3_put_does_not_follow_redirects_from_a_following_http_client() -> None:
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(
            307 if request.url.host == "bucket.example" else 200,
            headers={"Location": "https://redirected.example/object"} if request.url.host == "bucket.example" else {},
        )
    )
    handler: Final = AsyncHTTPHandler(transport=transport, follow_redirects=True)

    response: Final = await HttpxObjectStore(handler).put("https://bucket.example/object", b"body", {}, 1.0)
    await handler.close()

    assert (response.status_code, str(response.request.url)) == (307, "https://bucket.example/object")


def _sign(url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]:
    return {**headers, "Authorization": f"AWS4-HMAC-SHA256 test {len(content)}"}


def _append_record(state: tuple[ChainState, Batch], index: int) -> tuple[ChainState, Batch]:
    previous, records = state
    event: Final = AuditEvent(T0, "alice", None, "eu", "POST", f"/key/{index}", "allowed", 200, None, None)
    next_state, record = chain(event, Product("t"), previous)
    return next_state, (*records, record)


def _batch(count: int) -> Batch:
    return reduce(_append_record, range(count), (ChainState(), ()))[1]


@pytest.mark.asyncio
async def test_a_batch_is_one_locked_ndjson_object_under_a_dated_key_that_is_never_overwritten() -> None:
    store: Final = FakeStore([httpx.Response(200)])
    batch: Final = _batch(3)
    sink: Final = S3ObjectLockSink(SETTINGS, store, _sign, clock=lambda: T0)
    assert await sink.deliver(batch) == Delivered(3)
    ((url, content, headers, timeout),) = store.puts
    suffix: Final = batch[-1].hash[-12:]
    assert (
        url == f"https://audit.s3.eu-west-1.amazonaws.com/cp/{sink.run_id}/2026/10/13/12/"
        f"000000000001-000000000003-{suffix}.ndjson"
    )
    assert [json.loads(line) for line in content.decode().splitlines()] == [record.body for record in batch]
    assert headers["Authorization"] == f"AWS4-HMAC-SHA256 test {len(content)}" and timeout == 5.0
    assert headers["If-None-Match"] == "*" and headers["Content-Type"] == "application/x-ndjson"
    assert headers["Content-MD5"] == base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode()
    assert headers["x-amz-content-sha256"] == hashlib.sha256(content).hexdigest()
    assert (headers["x-amz-object-lock-mode"], headers["x-amz-object-lock-retain-until-date"]) == (
        "COMPLIANCE",
        "2026-11-12T12:00:00Z",
    )


@pytest.mark.asyncio
async def test_without_a_lock_mode_there_are_no_lock_headers_and_a_custom_endpoint_is_path_style() -> None:
    store: Final = FakeStore([httpx.Response(200)])
    settings: Final = replace(SETTINGS, object_lock_mode=None, endpoint_url="https://minio.internal:9000/")
    sink: Final = S3ObjectLockSink(settings, store, _sign)
    assert await sink.deliver(_batch(1)) == Delivered(1)
    ((url, _, headers, _),) = store.puts
    assert url.startswith(f"https://minio.internal:9000/audit/cp/{sink.run_id}/2026/10/13/12/000000000001-")
    assert not [key for key in headers if key.startswith("x-amz-object-lock")]


@pytest.mark.asyncio
async def test_a_china_partition_region_gets_its_own_dns_suffix() -> None:
    store: Final = FakeStore([httpx.Response(200)])
    sink: Final = S3ObjectLockSink(replace(SETTINGS, region="cn-north-1"), store, _sign)
    await sink.deliver(_batch(1))
    ((url, _, _, _),) = store.puts
    assert url.startswith(f"https://audit.s3.cn-north-1.amazonaws.com.cn/cp/{sink.run_id}/")


@pytest.mark.asyncio
async def test_sink_keys_are_unique_per_sink_and_stable_across_retries() -> None:
    batch: Final = _batch(1)
    first_store: Final = FakeStore([httpx.Response(200), httpx.Response(200)])
    second_store: Final = FakeStore([httpx.Response(200)])
    first: Final = S3ObjectLockSink(SETTINGS, first_store, _sign)
    second: Final = S3ObjectLockSink(SETTINGS, second_store, _sign)

    await first.deliver(batch)
    await first.deliver(batch)
    await second.deliver(batch)

    first_urls: Final = tuple(url for url, _, _, _ in first_store.puts)
    ((second_url, _, _, _),) = second_store.puts
    assert UUID(first.run_id).version == UUID(second.run_id).version == 4
    assert first_urls[0] == first_urls[1] and first_urls[0] != second_url


@pytest.mark.asyncio
async def test_expired_retention_omits_lock_headers() -> None:
    store: Final = FakeStore([httpx.Response(200)])
    sink: Final = S3ObjectLockSink(SETTINGS, store, _sign, clock=lambda: T0 + timedelta(days=30))

    await sink.deliver(_batch(1))

    ((_, _, headers, _),) = store.puts
    assert not [key for key in headers if key.startswith("x-amz-object-lock")]


@pytest.mark.asyncio
async def test_dotted_bucket_uses_regional_path_style_url() -> None:
    store: Final = FakeStore([httpx.Response(200)])
    sink: Final = S3ObjectLockSink(replace(SETTINGS, bucket="audit.logs"), store, _sign)

    await sink.deliver(_batch(1))

    ((url, _, _, _),) = store.puts
    assert url.startswith(f"https://s3.eu-west-1.amazonaws.com/audit.logs/cp/{sink.run_id}/")


@pytest.mark.asyncio
async def test_an_existing_key_is_delivered_while_s3_errors_and_network_errors_are_retried() -> None:
    store: Final = FakeStore(
        [
            httpx.Response(412, text="<Error><Code>PreconditionFailed</Code></Error>"),
            httpx.Response(503, text="<Error><Code>SlowDown</Code></Error>"),
            httpx.ConnectError("boom"),
        ]
    )
    sink: Final = S3ObjectLockSink(SETTINGS, store, _sign)
    batch: Final = _batch(2)
    assert await sink.deliver(batch) == Delivered(2), "the object is already there from an unacknowledged attempt"
    slow: Final = await sink.deliver(batch)
    assert isinstance(slow, Undelivered) and slow.reason.startswith(
        f"HTTP 503 SlowDown putting cp/{sink.run_id}/2026/10/13/12/"
    )
    down: Final = await sink.deliver(batch)
    assert isinstance(down, Undelivered) and down.reason.startswith(f"ConnectError putting cp/{sink.run_id}/")


@pytest.mark.asyncio
async def test_a_signer_that_cannot_get_credentials_is_a_failed_delivery_not_a_dead_worker() -> None:
    def no_credentials(url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]:
        raise RuntimeError("no credential chain")

    store: Final = FakeStore([])
    sink: Final = S3ObjectLockSink(SETTINGS, store, no_credentials)
    result: Final = await sink.deliver(_batch(1))
    assert isinstance(result, Undelivered) and result.reason.startswith(f"RuntimeError signing cp/{sink.run_id}/")
    assert store.puts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (301, "PermanentRedirect"),
        (301, "RequestTimeout"),
        (307, "TemporaryRedirect"),
        (400, "InvalidRequest"),
        (403, "AccessDenied"),
        (404, "NoSuchBucket"),
        (409, "Conflict"),
        (422, ""),
        (400, ""),
    ],
)
async def test_permanent_s3_failures_release_the_batch_and_count_loss(status: int, code: str) -> None:
    store: Final = FakeStore([httpx.Response(status, text=f"<Error><Code>{code}</Code></Error>"), httpx.Response(200)])
    sink: Final = S3ObjectLockSink(SETTINGS, store, _sign)
    pipeline: Final = build_pipeline((sink,), queue_size=3, batch_size=2, strict=False)
    first, second = _batch(2)
    assert pipeline.publish(first) == pipeline.publish(second) == (Accepted(),)
    (worker,) = pipeline.workers

    assert await worker.flush_once() == 2
    assert worker.rejected == 2 and worker.inflight is None and worker.queue.empty()
    assert pipeline.publish(first) == (Accepted(),)
    assert await worker.flush_once() == 1
    assert worker.rejected == 2 and worker.inflight is None and worker.queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (408, ""),
        (429, ""),
        (500, "InternalError"),
        (503, "SlowDown"),
        (400, "RequestTimeout"),
        (400, "RequestTimeoutException"),
        (400, "SlowDown"),
        (409, "ConditionalRequestConflict"),
        (409, "OperationAborted"),
    ],
)
async def test_transient_s3_failures_remain_retryable(status: int, code: str) -> None:
    store: Final = FakeStore([httpx.Response(status, text=f"<Error><Code>{code}</Code></Error>")])
    batch: Final = _batch(2)
    result: Final = await S3ObjectLockSink(SETTINGS, store, _sign).deliver(batch)
    assert isinstance(result, Undelivered)
    assert result.reason.startswith(f"HTTP {status} {code or 'no error code'} putting cp/")


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["ExpiredToken", "ExpiredTokenException", "RequestExpired", "TokenRefreshRequired"])
@pytest.mark.parametrize("status", [400, 403])
async def test_expired_s3_credentials_retry_the_same_batch_with_a_new_signature(status: int, code: str) -> None:
    signatures: Final = iter(("expired-local-credential", "refreshed-local-credential"))

    def sign(url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]:
        return {**headers, "Authorization": next(signatures)}

    store: Final = FakeStore([httpx.Response(status, text=f"<Error><Code>{code}</Code></Error>"), httpx.Response(200)])
    sink: Final = S3ObjectLockSink(SETTINGS, store, sign)
    pipeline: Final = build_pipeline((sink,), queue_size=2, batch_size=2, strict=False)
    (worker,) = pipeline.workers
    first, second = _batch(2)
    assert pipeline.publish(first) == pipeline.publish(second) == (Accepted(),)
    assert await worker.flush_once() == 2
    assert worker.rejected == 0 and worker.inflight is None and worker.queue.empty()
    before, after = store.puts
    assert before[:2] == after[:2]
    assert before[2]["Authorization"] == "expired-local-credential"
    assert after[2]["Authorization"] == "refreshed-local-credential"


@pytest.mark.asyncio
async def test_a_permanent_s3_reply_reports_the_rejected_count() -> None:
    sink: Final = S3ObjectLockSink(SETTINGS, FakeStore([httpx.Response(403)]), _sign)
    result: Final = await sink.deliver(_batch(2))
    assert isinstance(result, Rejected) and result.count == 2 and result.reason.startswith("HTTP 403 no error code")
