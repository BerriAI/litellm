import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, Undelivered
from litellm.proxy.control_plane.audit.s3 import S3ObjectLockSink
from litellm.proxy.control_plane.settings import S3SinkSettings

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
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
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _sign(url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]:
    return {**headers, "Authorization": f"AWS4-HMAC-SHA256 test {len(content)}"}


def _batch(count: int) -> Batch:
    state: Final[list[ChainState]] = [ChainState()]
    out: Final[list[AuditRecord]] = []
    for index in range(count):
        event = AuditEvent(T0, "alice", None, "eu", "POST", f"/key/{index}", "allowed", 200, None, None)
        state[0], record = chain(event, Product("t"), state[0])
        out.append(record)
    return tuple(out)


@pytest.mark.asyncio
async def test_a_batch_is_one_locked_ndjson_object_under_a_dated_key_that_is_never_overwritten() -> None:
    store = FakeStore([httpx.Response(200)])
    batch = _batch(3)
    assert await S3ObjectLockSink(SETTINGS, store, _sign).deliver(batch) == Delivered(3)
    ((url, content, headers, timeout),) = store.puts
    suffix = batch[-1].hash[-12:]
    assert url == f"https://audit.s3.eu-west-1.amazonaws.com/cp/2026/10/13/12/000000000001-000000000003-{suffix}.ndjson"
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
    store = FakeStore([httpx.Response(200)])
    settings = replace(SETTINGS, object_lock_mode=None, endpoint_url="https://minio.internal:9000/")
    assert await S3ObjectLockSink(settings, store, _sign).deliver(_batch(1)) == Delivered(1)
    ((url, _, headers, _),) = store.puts
    assert url.startswith("https://minio.internal:9000/audit/cp/2026/10/13/12/000000000001-000000000001-")
    assert not [key for key in headers if key.startswith("x-amz-object-lock")]


@pytest.mark.asyncio
async def test_a_china_partition_region_gets_its_own_dns_suffix() -> None:
    store = FakeStore([httpx.Response(200)])
    await S3ObjectLockSink(replace(SETTINGS, region="cn-north-1"), store, _sign).deliver(_batch(1))
    ((url, _, _, _),) = store.puts
    assert url.startswith("https://audit.s3.cn-north-1.amazonaws.com.cn/cp/")


@pytest.mark.asyncio
async def test_an_existing_key_is_delivered_while_s3_errors_and_network_errors_are_retried() -> None:
    store = FakeStore(
        [
            httpx.Response(412, text="<Error><Code>PreconditionFailed</Code></Error>"),
            httpx.Response(503, text="<Error><Code>SlowDown</Code></Error>"),
            httpx.ConnectError("boom"),
        ]
    )
    sink = S3ObjectLockSink(SETTINGS, store, _sign)
    batch = _batch(2)
    assert await sink.deliver(batch) == Delivered(2), "the object is already there from an unacknowledged attempt"
    slow = await sink.deliver(batch)
    assert isinstance(slow, Undelivered) and slow.reason.startswith("HTTP 503 SlowDown putting cp/2026/10/13/12/")
    down = await sink.deliver(batch)
    assert isinstance(down, Undelivered) and down.reason.startswith("ConnectError putting cp/")


@pytest.mark.asyncio
async def test_a_signer_that_cannot_get_credentials_is_a_failed_delivery_not_a_dead_worker() -> None:
    def no_credentials(url: str, content: bytes, headers: Mapping[str, str]) -> Mapping[str, str]:
        raise RuntimeError("no credential chain")

    store = FakeStore([])
    result = await S3ObjectLockSink(SETTINGS, store, no_credentials).deliver(_batch(1))
    assert isinstance(result, Undelivered) and result.reason.startswith("RuntimeError signing cp/")
    assert store.puts == []
