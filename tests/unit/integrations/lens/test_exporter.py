import asyncio
import json
import logging
from collections.abc import Mapping
from functools import partial
from itertools import chain
from typing import Final

import httpx
import prometheus_client
import pytest
from prometheus_client import CollectorRegistry, Counter
from pydantic import TypeAdapter

from litellm.integrations.lens import exporter as exporter_module
from litellm.integrations.lens.ledger import LedgerHandoff
from litellm.integrations.lens.exporter import (
    MAX_BUFFER_EVENTS,
    MAX_EVENT_BYTES,
    ExportFailure,
    LensExporter,
    encode_record,
    lens_export_counters,
)


def _confirmed(client: httpx.AsyncClient, request_id: str) -> LensExporter:
    return LensExporter(client, ledger=LedgerHandoff(), ledger_key=lambda kwargs, response: request_id)


def _commit(request_id: str, spend: float, api_key: str = "", team_id: str = "") -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "delivery_id": f"delivery-{request_id}",
        "spend": spend,
        "api_key": api_key,
        "team_id": team_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True), ids=("success-event", "failure-event"))
@pytest.mark.parametrize(
    ("tags", "expected"),
    (
        pytest.param((7,), ("7",), id="numeric"),
        pytest.param(("env:prod", 0, -7, 2.5, True, None), ("env:prod", "0", "-7", "2.5", "True", "None"), id="mixed"),
        pytest.param(("env:prod", "agent:research"), ("env:prod", "agent:research"), id="strings"),
    ),
)
async def test_request_tags_are_normalized_without_dropping_the_record(
    failed: bool, tags: tuple[object, ...], expected: tuple[str, ...]
) -> None:
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = _confirmed(client, "tagged-request")
        exporter.start()
        callback: Final = exporter.async_log_failure_event if failed else exporter.async_log_success_event
        try:
            await callback(
                {
                    "response_cost": 0.12,
                    "standard_logging_object": {
                        "id": "tagged-request",
                        "status": "failure" if failed else "success",
                        "response_cost": 0.12,
                        "request_tags": list(tags),
                    },
                },
                None,
                None,
                None,
            )
            exporter.ledger_committed((_commit("tagged-request", 0.12),))
        finally:
            await exporter.aclose()
    assert exporter.rows_written == 1
    assert exporter.rows_dropped == 0
    assert requests.qsize() == 1
    request: Final = requests.get_nowait()
    rows: Final = json.loads(request.content)
    assert request.url.path == "/internal/spend"
    assert len(rows) == 1
    assert rows[0]["request_id"] == "tagged-request"
    assert rows[0]["status"] == ("failure" if failed else "success")
    assert rows[0]["spend"] == 0.12
    assert rows[0]["request_tags"] == list(expected)


@pytest.mark.asyncio
async def test_request_export_ignores_unrelated_model_metadata_and_preserves_billing() -> None:
    received: Final = asyncio.Future[httpx.Request]()

    async def accept(request: httpx.Request) -> httpx.Response:
        received.set_result(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens.test/prefix/", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = _confirmed(client, "response-test")
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
        try:
            exporter.ledger_committed((_commit("response-test", 0.12, api_key="hash-test", team_id="team-test"),))
            request: Final = await asyncio.wait_for(received, timeout=5)
        finally:
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


@pytest.mark.parametrize(
    "value",
    ["a" * MAX_EVENT_BYTES, "界" * (MAX_EVENT_BYTES // 2), "\x01" * (MAX_EVENT_BYTES // 4)],
    ids=("oversized-ascii", "oversized-unicode", "escaped-json-expansion"),
)
def test_oversized_event_is_rejected_before_queueing(value: str) -> None:
    assert encode_record({"messages": value}) is ExportFailure.TOO_LARGE


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (429, 502, 503, 504, 0))
async def test_transient_failures_retry_the_same_batch_and_recover(status: int) -> None:
    requests: Final = asyncio.Queue[bytes]()
    waits: Final = asyncio.Queue[float]()

    async def retry_delay(seconds: float) -> None:
        waits.put_nowait(seconds)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request.content)
        if requests.qsize() == 3:
            return httpx.Response(204)
        if status == 0:
            raise httpx.ConnectError("private storage host", request=request)
        return httpx.Response(status)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(respond)) as client:
        exporter: Final = LensExporter(client, sleep=retry_delay)
        assert exporter.enqueue(b'{"id":1}')
        exporter.start()
        await exporter.aclose()
    assert tuple(requests.get_nowait() for _ in range(3)) == (b'[{"id":1}]',) * 3
    assert tuple(waits.get_nowait() for _ in range(2)) == (1.0, 2.0)
    assert (exporter.rows_written, exporter.rows_dropped, exporter.buffered_bytes, exporter.buffered_events) == (
        1,
        0,
        0,
        0,
    )
    assert exporter.last_error == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,attempts,reason", ((401, 1, "HTTP 401"), (500, 1, "HTTP 500"), (503, 3, "retry limit reached"))
)
async def test_failed_exports_are_counted_and_release_all_buffer_capacity(
    status: int, attempts: int, reason: str
) -> None:
    requests: Final = asyncio.Queue[bytes]()

    async def no_wait(_: float) -> None:
        return None

    def reject(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request.content)
        return httpx.Response(status, text="private credentials")

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(reject)) as client:
        exporter: Final = LensExporter(client, sleep=no_wait)
        assert exporter.enqueue(b"{}")
        exporter.start()
        exporter.start()
        await exporter.aclose()
    assert requests.qsize() == attempts
    assert (exporter.rows_written, exporter.rows_dropped, exporter.buffered_bytes, exporter.buffered_events) == (
        0,
        1,
        0,
        0,
    )
    assert exporter.last_error == reason
    assert not exporter.enqueue(b"{}")
    assert exporter.rows_dropped == 2


@pytest.mark.asyncio
async def test_cancelled_inflight_export_drops_the_batch_and_pending_records() -> None:
    started: Final = asyncio.Event()

    async def block(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Future[None]()
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(block)) as client:
        exporter: Final = LensExporter(client)
        assert exporter.enqueue(b"{}")
        exporter.start()
        await started.wait()
        assert exporter.enqueue(b"{}")
        assert exporter.task is not None
        exporter.task.cancel()
        await exporter.aclose()
    assert (exporter.rows_written, exporter.rows_dropped, exporter.buffered_bytes, exporter.buffered_events) == (
        0,
        2,
        0,
        0,
    )


@pytest.mark.asyncio
async def test_byte_budget_rejects_large_queue_and_shutdown_without_start_discards_it() -> None:
    from litellm.integrations.lens.exporter import MAX_BUFFER_BYTES

    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporter: Final = LensExporter(client)
        assert not exporter.enqueue(b"x" * (MAX_EVENT_BYTES + 1))
        for _ in range(MAX_BUFFER_BYTES // MAX_EVENT_BYTES):
            assert exporter.enqueue(b"x" * MAX_EVENT_BYTES)
        assert not exporter.enqueue(b"x")
        assert exporter.buffered_bytes == MAX_BUFFER_BYTES
        await exporter.aclose()
    assert exporter.rows_dropped == MAX_BUFFER_BYTES // MAX_EVENT_BYTES + 2
    assert (exporter.buffered_bytes, exporter.buffered_events) == (0, 0)


@pytest.mark.asyncio
async def test_batches_stay_bounded_without_losing_or_reordering_records() -> None:
    from litellm.integrations.lens.exporter import MAX_BATCH_BYTES

    bodies: Final = asyncio.Queue[bytes]()

    def accept(request: httpx.Request) -> httpx.Response:
        bodies.put_nowait(request.content)
        return httpx.Response(204)

    records: Final = tuple(encode_record({"id": index, "text": "x" * (MAX_EVENT_BYTES // 2)}) for index in range(10))
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client)
        for record in records:
            assert isinstance(record, bytes)
            assert exporter.enqueue(record)
        exporter.start()
        await exporter.aclose()
    sent: Final = tuple(bodies.get_nowait() for _ in range(bodies.qsize()))
    assert len(sent) == 2
    assert all(len(body) <= MAX_BATCH_BYTES for body in sent)
    assert [row["id"] for row in chain.from_iterable(json.loads(body) for body in sent)] == list(range(10))
    assert exporter.rows_written == 10
    assert exporter.rows_dropped == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        None,
        {"id": []},
        {"id": "x", "messages": [{"content": "x" * MAX_EVENT_BYTES}]},
        {"id": "x", "messages": [{"content": "\x01" * (MAX_EVENT_BYTES // 4)}]},
        {"id": "x", "response_cost": float("nan")},
    ),
    ids=("missing", "invalid-id", "oversized-input", "escaped-row-expansion", "invalid-row-number"),
)
async def test_invalid_callback_data_never_interrupts_model_requests(payload: object) -> None:
    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporter: Final = LensExporter(client)
        await exporter.async_log_failure_event({"standard_logging_object": payload}, None, None, None)
        await exporter.aclose()
    assert exporter.rows_written == 0
    assert exporter.buffered_bytes == 0
    assert exporter.rows_dropped == (0 if payload is None else 1)


def test_serialization_rejects_recursive_payloads() -> None:
    cyclic: Final[dict[str, object]] = {}  # mutable-ok: deliberately constructs a cyclic callback payload
    cyclic["self"] = cyclic
    assert encode_record(cyclic) is ExportFailure.TOO_LARGE


@pytest.mark.parametrize("value", (float("nan"), object(), "\ud800"), ids=("nan", "unsupported", "surrogate"))
def test_serialization_returns_a_failure_for_invalid_payloads(value: object) -> None:
    assert encode_record({"value": value}) is ExportFailure.INVALID


@pytest.mark.asyncio
async def test_written_and_dropped_rows_are_exported_as_prometheus_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry: Final = CollectorRegistry()
    # Exercise the real factory without sharing its cached collectors with other tests.
    monkeypatch.setattr(prometheus_client, "Counter", partial(Counter, registry=registry))
    monkeypatch.setattr(exporter_module, "lens_export_counters", lens_export_counters.__wrapped__)
    statuses: Final = iter((204, 500))
    answered: Final = asyncio.Queue[int]()

    def answer(request: httpx.Request) -> httpx.Response:
        status: Final = next(statuses)
        answered.put_nowait(status)
        return httpx.Response(status)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(answer)) as client:
        exporter: Final = LensExporter(
            client, ledger=LedgerHandoff(), ledger_key=lambda kwargs, response: str(kwargs["call"])
        )
        exporter.start()
        try:
            for request_id in ("accepted", "rejected"):
                await exporter.async_log_success_event(
                    {"call": request_id, "standard_logging_object": {"id": request_id, "response_cost": 0.1}},
                    None,
                    None,
                    None,
                )
                exporter.ledger_committed((_commit(request_id, 0.1),))
                await asyncio.wait_for(answered.get(), timeout=5)
        finally:
            await exporter.aclose()
    assert (exporter.rows_written, exporter.rows_dropped) == (1, 1)
    assert registry.get_sample_value("litellm_lens_export_rows_written_total") == exporter.rows_written
    assert registry.get_sample_value("litellm_lens_export_rows_dropped_total") == exporter.rows_dropped


@pytest.mark.asyncio
async def test_a_rejecting_store_is_logged_as_an_error_naming_lens(caplog: pytest.LogCaptureFixture) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(reject)) as client:
            exporter: Final = LensExporter(client)
            assert exporter.enqueue(b"{}")
            exporter.start()
            await exporter.aclose()
    failures: Final = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert len(failures) == 1
    assert "lens" in failures[0] and "HTTP 500" in failures[0]


def _ledger_event(request_id: str) -> dict[str, object]:
    return {
        "response_cost": 9.0,
        "standard_logging_object": {"id": request_id, "status": "success", "response_cost": 9.0},
    }


@pytest.mark.asyncio
async def test_ledger_backed_records_are_sent_only_after_their_commit_and_off_the_writer_path() -> None:
    from litellm.integrations.lens.ledger import LedgerHandoff

    received: Final = asyncio.Future[bytes]()

    def accept(request: httpx.Request) -> httpx.Response:
        received.set_result(request.content)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client, ledger=LedgerHandoff(), ledger_key=lambda kwargs, response: "ledger-a")
        exporter.start()
        try:
            await exporter.async_log_success_event(_ledger_event("ledger-a"), None, None, None)
            assert (exporter.buffered_events, received.done()) == (0, False)
            exporter.ledger_committed(
                ({"request_id": "ledger-a", "delivery_id": "d", "spend": 0.5, "prompt_tokens": 1},)
            )
            assert (exporter.buffered_events, len(exporter.commits)) == (0, 1)
            body: Final = await asyncio.wait_for(received, timeout=5)
        finally:
            await exporter.aclose()
    rows: Final = TypeAdapter(list[dict[str, object]]).validate_json(body)
    assert (rows[0]["request_id"], rows[0]["spend"], rows[0]["prompt_tokens"], rows[0]["total_tokens"]) == (
        "ledger-a",
        0.5,
        1,
        1,
    )
    assert (exporter.rows_written, exporter.rows_dropped) == (1, 0)


@pytest.mark.asyncio
async def test_confirmed_deliveries_beyond_the_pending_budget_are_counted_as_dropped() -> None:
    from litellm.integrations.lens import exporter as module
    from litellm.integrations.lens.ledger import LedgerHandoff

    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporter: Final = LensExporter(client, ledger=LedgerHandoff(), ledger_key=lambda kwargs, response: "k")
        try:
            batch: Final = tuple({"request_id": f"r{index}", "delivery_id": f"d{index}"} for index in range(100))
            for _ in range(module.MAX_PENDING_COMMITS // 100):
                exporter.ledger_committed(batch)
            exporter.ledger_committed(batch[:1])
            assert (len(exporter.commits), exporter.rows_dropped) == (module.MAX_PENDING_COMMITS, 1)
        finally:
            await exporter.aclose()


@pytest.mark.asyncio
async def test_records_without_a_confirmed_ledger_commit_are_counted_and_never_sent() -> None:
    from litellm.integrations.lens.ledger import LedgerHandoff

    def unknown(kwargs: object, response: object) -> str | None:
        raise RuntimeError("settings unavailable")

    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporters: Final = (
            LensExporter(client),
            LensExporter(client, ledger=LedgerHandoff(), ledger_key=lambda kwargs, response: None),
            LensExporter(client, ledger=LedgerHandoff(), ledger_key=unknown),
        )
        try:
            for exporter in exporters:
                await exporter.async_log_success_event(_ledger_event("unconfirmed"), None, None, None)
                exporter.ledger_committed((_commit("unconfirmed", 0.5),))
        finally:
            for exporter in exporters:
                await exporter.aclose()
    assert [(exporter.rows_written, exporter.buffered_events) for exporter in exporters] == [(0, 0)] * 3
    assert [exporter.rows_dropped for exporter in exporters] == [1, 2, 2]


@pytest.mark.asyncio
async def test_a_record_the_committed_owner_grows_past_the_event_limit_is_counted_without_blocking_others() -> None:
    sent: Final = asyncio.Queue[bytes]()

    def accept(request: httpx.Request) -> httpx.Response:
        sent.put_nowait(request.content)
        return httpx.Response(204)

    grown: Final = {
        "response_cost": 9.0,
        "litellm_call_id": "grown",
        "standard_logging_object": {
            "id": "grown",
            "status": "success",
            "response_cost": 9.0,
            "metadata": {"padding": "x" * (MAX_EVENT_BYTES - 2000)},
        },
    }
    owner: Final = {field: "a" * 512 for field in ("api_key", "user", "team_id", "organization_id", "end_user")}
    handoff: Final = LedgerHandoff()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(
            client, ledger=handoff, ledger_key=lambda kwargs, response: str(kwargs.get("litellm_call_id"))
        )
        exporter.start()
        try:
            await exporter.async_log_success_event(grown, None, None, None)
            assert (len(handoff.held), exporter.rows_dropped, sent.empty()) == (1, 0, True)
            exporter.ledger_committed(({"request_id": "grown", "delivery_id": "d-grown", "spend": 0.5, **owner},))
            await exporter.async_log_success_event(
                {**_ledger_event("after"), "litellm_call_id": "after"}, None, None, None
            )
            exporter.ledger_committed(({"request_id": "after", "delivery_id": "d-after", "spend": 0.25},))
            body: Final = await asyncio.wait_for(sent.get(), timeout=5)
        finally:
            await exporter.aclose()
    rows: Final = TypeAdapter(list[dict[str, object]]).validate_json(body)
    assert [(row["request_id"], row["spend"]) for row in rows] == [("after", 0.25)]
    assert (exporter.rows_written, exporter.rows_dropped, sent.empty(), len(handoff.held)) == (1, 1, True, 0)
