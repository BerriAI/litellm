"""Regression tests for draining s3_v2 callback queues on proxy shutdown (#52588)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.unit.integrations.s3_v2_rust_support import (
    ScriptedBucket,
    log_payload,
    scripted_bucket,
)

import litellm
import litellm.litellm_core_utils.litellm_logging as litellm_logging
import litellm.proxy.management_helpers.audit_logs as audit_logs
import litellm.proxy.proxy_server as ps
from litellm.integrations import s3_v2
from litellm.integrations.s3_v2 import S3Logger, collect_active_s3_v2_loggers, flush_s3_v2_loggers
from litellm.litellm_core_utils.logging_worker import LoggingWorker
from litellm.proxy.management_helpers.audit_logs import track_audit_task
from litellm.proxy.proxy_server import proxy_shutdown_event
from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager
from litellm.responses.streaming_iterator import spawn_logging_task
from litellm.rust_bridge import _native
from litellm.types.utils import StandardAuditLogPayload

pytestmark = pytest.mark.skipif(not hasattr(_native, "NativeS3Log"), reason="the built _native has no NativeS3Log")

START: Final = datetime(2026, 11, 18, 12, 0, 0)


def _make_s3_logger(
    bucket: ScriptedBucket, *, bucket_name: str = "synthetic-logs", flush_interval: float = 3600.0
) -> S3Logger:
    return S3Logger(
        s3_callback_params_override={
            "s3_bucket_name": bucket_name,
            "s3_endpoint_url": bucket.url,
            "s3_region_name": "us-east-1",
            "s3_aws_access_key_id": "synthetic-access",
            "s3_aws_secret_access_key": "synthetic-secret",
        },
        s3_flush_interval=flush_interval,
    )


def _isolate_shutdown_dependencies(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock, MagicMock]:
    fake_cache = MagicMock()
    fake_cache.disconnect = AsyncMock()
    fake_jwt = MagicMock()
    fake_jwt.close = AsyncMock()
    fake_db_writer = MagicMock()
    fake_db_writer.close = AsyncMock()

    monkeypatch.setattr(ps, "prisma_client", None, raising=False)
    monkeypatch.setattr(ps, "db_writer_client", fake_db_writer, raising=False)
    monkeypatch.setattr(ps, "jwt_http_handler", fake_jwt, raising=False)
    monkeypatch.setattr(ps, "shutdown_billing_metrics_recorder", None, raising=False)
    monkeypatch.setattr(ps, "drain_post_call_redis_batches", AsyncMock(), raising=False)
    monkeypatch.setattr(litellm, "cache", fake_cache, raising=False)
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "failure_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_failure_callback", [], raising=False)
    monkeypatch.setattr(litellm, "input_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_input_callback", [], raising=False)
    monkeypatch.setattr(litellm, "service_callback", [], raising=False)
    monkeypatch.setattr(litellm, "audit_log_callbacks", [], raising=False)
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [], raising=False)
    monkeypatch.setattr(audit_logs, "_audit_log_callback_cache", {}, raising=False)
    GracefulShutdownManager.reset()
    return fake_cache, fake_jwt, fake_db_writer


@pytest.fixture
def bucket() -> Iterator[ScriptedBucket]:
    with scripted_bucket() as scripted:
        yield scripted


@pytest.mark.asyncio
async def test_proxy_shutdown_drains_queued_s3_v2_event_and_deduplicates_registrations(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    _isolate_shutdown_dependencies(monkeypatch)
    logger = _make_s3_logger(bucket)
    await log_payload(logger, {"id": "shutdown-event-1", "metadata": {}, "messages": [], "response_cost": 0.01}, START)
    assert len(logger.sink.queued_keys()) == 1

    # Register the same instance across every list and cache to verify deduplication.
    monkeypatch.setattr(litellm, "callbacks", [logger], raising=False)
    monkeypatch.setattr(litellm, "success_callback", [logger], raising=False)
    monkeypatch.setattr(litellm, "failure_callback", [logger], raising=False)
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [logger], raising=False)
    monkeypatch.setattr(audit_logs, "_audit_log_callback_cache", {"s3_v2": logger}, raising=False)

    assert collect_active_s3_v2_loggers() == (logger,)

    flush_calls = 0
    original_flush_for_shutdown = logger.flush_for_shutdown

    async def counting_flush() -> bool:
        nonlocal flush_calls
        flush_calls += 1
        return await original_flush_for_shutdown()

    monkeypatch.setattr(logger, "flush_for_shutdown", counting_flush)

    await asyncio.wait_for(proxy_shutdown_event(), timeout=5.0)

    puts = bucket.puts()
    assert flush_calls == 1
    assert len(puts) == 1
    assert json.loads(puts[0].body)["id"] == "shutdown-event-1"
    assert logger.sink.queued_keys() == []
    assert logger.periodic_flush_task.done()


@pytest.mark.asyncio
async def test_proxy_shutdown_drains_multiple_s3_v2_instances_across_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_shutdown_dependencies(monkeypatch)
    with scripted_bucket() as primary_bucket, scripted_bucket() as audit_bucket:
        primary_logger = _make_s3_logger(primary_bucket, bucket_name="primary-logs")
        audit_logger = _make_s3_logger(audit_bucket, bucket_name="audit-logs")

        await log_payload(primary_logger, {"id": "primary-evt", "metadata": {}, "messages": []}, START)
        audit_payload: StandardAuditLogPayload = {
            "id": "audit-evt",
            "updated_at": "2026-11-18T12:00:00+00:00",
            "changed_by": "admin",
            "changed_by_api_key": "sk-admin",
            "action": "create",
            "table_name": "LiteLLM_VerificationToken",
            "object_id": "tok-1",
            "before_value": None,
            "updated_values": None,
        }
        await audit_logger.async_log_audit_log_event(audit_payload)

        monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [primary_logger], raising=False)
        monkeypatch.setattr(audit_logs, "_audit_log_callback_cache", {"s3_v2": audit_logger}, raising=False)

        await asyncio.wait_for(proxy_shutdown_event(), timeout=5.0)

        primary_puts = primary_bucket.puts()
        audit_puts = audit_bucket.puts()
        assert len(primary_puts) == 1
        assert json.loads(primary_puts[0].body)["id"] == "primary-evt"
        assert len(audit_puts) == 1
        assert json.loads(audit_puts[0].body)["id"] == "audit-evt"
        assert primary_logger.sink.queued_keys() == []
        assert audit_logger.sink.queued_keys() == []
        assert primary_logger.periodic_flush_task.done()
        assert audit_logger.periodic_flush_task.done()


@pytest.mark.asyncio
async def test_proxy_shutdown_drains_pending_producers_before_s3_v2_flush(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    _isolate_shutdown_dependencies(monkeypatch)
    isolated_worker = LoggingWorker()
    monkeypatch.setattr(ps, "GLOBAL_LOGGING_WORKER", isolated_worker, raising=False)

    logger = _make_s3_logger(bucket)
    monkeypatch.setattr(litellm, "callbacks", [logger], raising=False)

    async def produce_audit() -> None:
        await asyncio.sleep(0.02)
        audit_payload: StandardAuditLogPayload = {
            "id": "from-audit-task",
            "updated_at": "2026-11-18T12:00:00+00:00",
            "changed_by": "admin",
            "changed_by_api_key": "sk-admin",
            "action": "update",
            "table_name": "LiteLLM_TeamTable",
            "object_id": "team-1",
            "before_value": None,
            "updated_values": None,
        }
        await logger.async_log_audit_log_event(audit_payload)

    async def produce_stream_log() -> None:
        await asyncio.sleep(0.02)
        await log_payload(logger, {"id": "from-stream-task", "metadata": {}, "messages": []}, START)

    async def produce_worker_log() -> None:
        await asyncio.sleep(0.02)
        await log_payload(logger, {"id": "from-worker-task", "metadata": {}, "messages": []}, START)

    track_audit_task(asyncio.create_task(produce_audit()))
    spawn_logging_task(asyncio.get_running_loop(), produce_stream_log(), task_name="s3-test-stream-producer")
    isolated_worker.ensure_initialized_and_enqueue(produce_worker_log())

    try:
        await asyncio.wait_for(proxy_shutdown_event(), timeout=5.0)
    finally:
        await isolated_worker.stop(discard_pending=True, timeout=1.0)

    uploaded_ids = {json.loads(put.body)["id"] for put in bucket.puts()}
    assert uploaded_ids == {"from-audit-task", "from-stream-task", "from-worker-task"}
    assert logger.sink.queued_keys() == []


@pytest.mark.asyncio
async def test_proxy_shutdown_flushes_s3_v2_before_closing_cache_jwt_and_db_writer(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    fake_cache, fake_jwt, fake_db_writer = _isolate_shutdown_dependencies(monkeypatch)
    events: list[str] = []

    fake_cache.disconnect = AsyncMock(side_effect=lambda: events.append("cache_disconnect"))
    fake_jwt.close = AsyncMock(side_effect=lambda: events.append("jwt_close"))
    fake_db_writer.close = AsyncMock(side_effect=lambda: events.append("db_writer_close"))
    monkeypatch.setattr(ps, "cleanup_router_config_variables", lambda: events.append("cleanup_router"), raising=False)

    logger = _make_s3_logger(bucket)
    await log_payload(logger, {"id": "ordered-evt", "metadata": {}, "messages": []}, START)
    monkeypatch.setattr(litellm, "callbacks", [logger], raising=False)

    original_flush_for_shutdown = logger.flush_for_shutdown

    async def ordered_flush() -> bool:
        ok = await original_flush_for_shutdown()
        events.append("s3_flushed")
        return ok

    monkeypatch.setattr(logger, "flush_for_shutdown", ordered_flush)

    await asyncio.wait_for(proxy_shutdown_event(), timeout=5.0)

    assert events == ["s3_flushed", "cache_disconnect", "jwt_close", "db_writer_close", "cleanup_router"]
    assert len(bucket.puts()) == 1


@pytest.mark.asyncio
async def test_proxy_shutdown_bounds_slow_s3_v2_flush_and_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_cache, fake_jwt, _fake_db_writer = _isolate_shutdown_dependencies(monkeypatch)
    monkeypatch.setattr(ps, "S3_SHUTDOWN_FLUSH_TIMEOUT_SECONDS", 0.15, raising=False)

    with scripted_bucket() as slow_bucket:
        logger = _make_s3_logger(slow_bucket)

        async def hung_flush() -> None:
            await asyncio.sleep(10.0)

        monkeypatch.setattr(logger, "flush_queue", hung_flush)
        monkeypatch.setattr(litellm, "callbacks", [logger], raising=False)

        started = time.monotonic()
        with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
            await asyncio.wait_for(proxy_shutdown_event(), timeout=2.0)
        elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert fake_jwt.close.await_count == 1
    assert logger.periodic_flush_task.done()
    assert any("S3 v2 shutdown flush incomplete" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_proxy_shutdown_reports_upload_failure_and_still_drains_healthy_sibling(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _isolate_shutdown_dependencies(monkeypatch)

    with scripted_bucket() as failing_bucket, scripted_bucket() as healthy_bucket:
        failing_bucket.reply("fail-evt", (503, ""))
        failing_logger = _make_s3_logger(failing_bucket, bucket_name="failing-logs")
        healthy_logger = _make_s3_logger(healthy_bucket, bucket_name="healthy-logs")

        await log_payload(failing_logger, {"id": "fail-evt", "metadata": {}, "messages": []}, START)
        await log_payload(healthy_logger, {"id": "healthy-evt", "metadata": {}, "messages": []}, START)

        monkeypatch.setattr(litellm, "callbacks", [failing_logger, healthy_logger], raising=False)
        monkeypatch.setattr(s3_v2._Host, "callback_failed", lambda _self: None)

        with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
            await asyncio.wait_for(proxy_shutdown_event(), timeout=15.0)

        assert len(healthy_bucket.puts()) == 1
        assert json.loads(healthy_bucket.puts()[0].body)["id"] == "healthy-evt"
        assert healthy_logger.sink.queued_keys() == []
        assert len(failing_logger.sink.queued_keys()) == 1
        assert failing_logger.periodic_flush_task.done()
        assert healthy_logger.periodic_flush_task.done()
        assert any("S3 v2 shutdown flush incomplete" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_flush_s3_v2_loggers_isolates_exceptions_across_instances(
    monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket
) -> None:
    _isolate_shutdown_dependencies(monkeypatch)
    broken_logger = _make_s3_logger(bucket, bucket_name="broken")
    healthy_logger = _make_s3_logger(bucket, bucket_name="healthy")
    await log_payload(healthy_logger, {"id": "sibling-survives", "metadata": {}, "messages": []}, START)

    monkeypatch.setattr(broken_logger, "flush_queue", AsyncMock(side_effect=RuntimeError("boom")))

    flushed = await flush_s3_v2_loggers(timeout_seconds=5.0, loggers=(broken_logger, healthy_logger))

    assert flushed is False
    assert len(bucket.puts()) == 1
    assert json.loads(bucket.puts()[0].body)["id"] == "sibling-survives"
    assert broken_logger.periodic_flush_task.done()
    assert healthy_logger.periodic_flush_task.done()
