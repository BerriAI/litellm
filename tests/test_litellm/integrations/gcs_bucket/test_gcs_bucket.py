import asyncio
import json
from typing import Any, Final
from unittest.mock import patch

import pytest

from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.integrations.gcs_bucket import GCSFlushResult, GCSLoggingConfig, GCSLogQueueItem
from litellm.types.utils import StandardLoggingPayload


class _FakeUploadGCSLogger(GCSBucketLogger):
    """Skips GCP auth; an upload raises when it carries any id in `failing_ids`, otherwise it is recorded"""

    def __init__(self, queue_maxsize: int = 0) -> None:
        with patch("litellm.proxy.proxy_server.premium_user", True):  # test-quality-ok: GCS logging is premium-gated
            super().__init__(bucket_name="test-bucket")
        self.log_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.failing_ids: frozenset[str] = frozenset()
        self.arriving_during_upload: tuple[str, ...] = ()
        self.uploaded: list[list[str]] = []

    async def enqueue(self, request_id: str) -> None:
        payload: Final = StandardLoggingPayload(id=request_id)  # pyright: ignore[reportCallIssue]  # partial payload is enough for queueing
        await self._enqueue(GCSLogQueueItem(payload=payload, kwargs={}, response_obj={"id": request_id}))

    def queued_ids(self) -> list[str]:
        return [self.log_queue.get_nowait()["payload"]["id"] for _ in range(self.log_queue.qsize())]

    async def get_gcs_logging_config(self, kwargs: dict[str, Any] | None = None) -> GCSLoggingConfig:
        return GCSLoggingConfig(bucket_name="test-bucket", vertex_instance=None, path_service_account=None)

    async def construct_request_headers(
        self, service_account_json: str | None, vertex_instance: VertexBase | None = None
    ) -> dict[str, str]:
        return {}

    async def _log_json_data_on_gcs(
        self, headers: dict[str, str], bucket_name: str, object_name: str, logging_payload: StandardLoggingPayload | str
    ) -> None:
        ids: Final = (
            [json.loads(line)["id"] for line in logging_payload.splitlines()]
            if isinstance(logging_payload, str)
            else [logging_payload["id"]]
        )
        for request_id in self.arriving_during_upload:
            await self.enqueue(request_id)
        if self.failing_ids.intersection(ids):
            raise RuntimeError("storage.googleapis.com returned 404")
        self.uploaded.append(ids)


@pytest.mark.asyncio
async def test_failed_batch_stays_queued_and_is_retried_on_the_next_flush():
    logger = _FakeUploadGCSLogger()
    logger.failing_ids = frozenset({"req-1"})
    await logger.enqueue("req-1")
    await logger.enqueue("req-2")

    failed_flush = await logger.flush_queue_and_report()

    assert failed_flush == GCSFlushResult(sent=0, failed=2)
    assert logger.log_queue.qsize() == 2
    assert logger.uploaded == []

    logger.failing_ids = frozenset()
    retried_flush = await logger.flush_queue_and_report()

    assert retried_flush == GCSFlushResult(sent=2, failed=0)
    assert logger.log_queue.qsize() == 0
    assert logger.uploaded == [["req-1", "req-2"]]


@pytest.mark.asyncio
async def test_individual_mode_requeues_only_the_failed_items():
    logger = _FakeUploadGCSLogger()
    logger.use_batched_logging = False
    logger.failing_ids = frozenset({"req-fail"})
    await logger.enqueue("req-ok")
    await logger.enqueue("req-fail")

    result = await logger.flush_queue_and_report()

    assert result == GCSFlushResult(sent=1, failed=1)
    assert logger.uploaded == [["req-ok"]]
    assert logger.queued_ids() == ["req-fail"]


@pytest.mark.asyncio
async def test_enqueue_on_a_full_queue_whose_flush_failed_drops_the_oldest_event():
    logger = _FakeUploadGCSLogger(queue_maxsize=2)
    logger.failing_ids = frozenset({"req-1", "req-2", "req-3"})
    await logger.enqueue("req-1")
    await logger.enqueue("req-2")

    await logger.enqueue("req-3")

    assert logger.queued_ids() == ["req-2", "req-3"]


@pytest.mark.asyncio
async def test_failed_batch_is_dropped_when_new_events_filled_the_queue_during_the_upload():
    logger = _FakeUploadGCSLogger(queue_maxsize=2)
    logger.failing_ids = frozenset({"req-1"})
    logger.arriving_during_upload = ("req-3", "req-4")
    await logger.enqueue("req-1")
    await logger.enqueue("req-2")

    result = await logger.flush_queue_and_report()

    assert result == GCSFlushResult(sent=0, failed=2)
    assert logger.queued_ids() == ["req-3", "req-4"]


@pytest.mark.asyncio
async def test_empty_queue_flush_reports_nothing_sent_or_failed():
    logger = _FakeUploadGCSLogger()

    assert await logger.flush_queue_and_report() == GCSFlushResult(sent=0, failed=0)
