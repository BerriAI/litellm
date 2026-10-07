import asyncio
import json
from typing import Final

import httpx
import pytest

from litellm.tracing.exporter import MAX_BUFFER_EVENTS, MAX_EVENT_BYTES, LensExporter, encode_record


@pytest.mark.asyncio
async def test_request_export_ignores_unrelated_model_metadata_and_preserves_billing() -> None:
    received: Final = asyncio.Future[httpx.Request]()

    async def accept(request: httpx.Request) -> httpx.Response:
        received.set_result(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens.test/prefix/", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client)
        exporter.start()
        await exporter.async_log_success_event(
            {
                "response_cost": 0.12,
                "standard_logging_object": {
                    "id": "response-test",
                    "status": "success",
                    "call_type": "acompletion",
                    "model": "test-model",
                    "response_cost": 0.12,
                    "model_map_information": {"model_map_value": {"extra_pricing_metadata": None}},
                    "metadata": {"user_api_key_hash": "hash-test", "user_api_key_team_id": "team-test"},
                    "messages": [{"role": "user", "content": "Check a refund"}],
                    "response": {"choices": [{"message": {"content": "Refund failed"}}]},
                },
            },
            None,
            None,
            None,
        )
        request: Final = await asyncio.wait_for(received, timeout=1)
        await exporter.aclose()
    rows: Final = json.loads(request.content)
    assert request.url.path == "/prefix/internal/spend"
    assert len(rows) == 1
    assert rows[0]["response_id"] == "response-test"
    assert rows[0]["spend"] == 0.12
    assert rows[0]["api_key"] == "hash-test"
    assert rows[0]["team_id"] == "team-test"
    assert json.loads(rows[0]["messages"]) == [{"role": "user", "content": "Check a refund"}]
    assert exporter.rows_written == 1
    assert exporter.rows_dropped == 0
    assert exporter.buffered_bytes == 0


@pytest.mark.asyncio
async def test_inflight_records_count_toward_the_queue_limit() -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def blocked(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens.test", transport=httpx.MockTransport(blocked)) as client:
        exporter: Final = LensExporter(client)
        for _ in range(MAX_BUFFER_EVENTS):
            assert exporter.enqueue(b"{}")
        exporter.start()
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not exporter.enqueue(b"{}")
        assert exporter.buffered_events == MAX_BUFFER_EVENTS
        assert exporter.rows_dropped == 1
        release.set()
        await exporter.aclose()
        assert exporter.rows_written == MAX_BUFFER_EVENTS
        assert exporter.buffered_events == 0
        assert exporter.buffered_bytes == 0


@pytest.mark.parametrize("value", ["a" * MAX_EVENT_BYTES, "界" * (MAX_EVENT_BYTES // 2)])
def test_oversized_event_is_rejected_before_queueing(value: str) -> None:
    with pytest.raises(OverflowError):
        encode_record({"messages": value})
