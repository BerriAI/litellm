import asyncio
import copy
import hashlib
import json
import logging
import threading
from collections.abc import Callable, Coroutine, Mapping
from datetime import datetime, timezone
from functools import partial
from itertools import chain
from typing import Final, TypeVar

import httpx
import prometheus_client
import pytest
from prometheus_client import CollectorRegistry, Counter
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.clickhouse.clickhouse_spend_logger import spend_log_row_from_payload
from litellm.integrations.clickhouse.spend_log_types import SpendLogPayload
from litellm.integrations.lens import exporter as exporter_module
from litellm.integrations.lens.content import (
    MAX_NATIVE_UPLOADS,
    ContentLimits,
    Owner,
    StoredReference,
    Unavailable,
    content_signing_key,
    signed_reference,
    verified_stored,
)
from litellm.integrations.lens.exporter import (
    MAX_BUFFER_EVENTS,
    MAX_EVENT_BYTES,
    ExportFailure,
    LensExporter,
    encode_record,
    lens_export_counters,
)
from litellm.integrations.lens.ledger import LedgerHandoff
from litellm.litellm_core_utils.core_helpers import as_str_mapping


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


_T = TypeVar("_T")
_KEY: Final = content_signing_key("test-service-token-with-32-characters")
_OWNER: Final = Owner(team_id="team-a", api_key="hash-a", request_id="request-a")
_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TEXT: Final = TypeAdapter(dict[str, str])
_SMALL_INLINE: Final = ContentLimits(inline_bytes=4096)


class _Store:
    def __init__(
        self, outcome: Callable[[], "asyncio.Future[bool]"] | None = None, prefix: str | None = "logs"
    ) -> None:
        self.outcome: Final = outcome
        self.prefix: Final = prefix
        self.uploads: Final = asyncio.Queue[tuple[str, str]]()
        self.natives: Final = asyncio.Queue["asyncio.Future[bool]"]()
        self.started: Final = asyncio.Event()

    def object_key(self, name: str) -> str | None:
        return None if self.prefix is None else f"{self.prefix}/{name}"

    def upload(self, key: str, body: str) -> "asyncio.Future[bool]":
        self.uploads.put_nowait((key, body))
        native: Final = self.outcome() if self.outcome is not None else _settled(True)
        self.natives.put_nowait(native)
        self.started.set()
        return native

    def download(self, key: str, max_bytes: int) -> "asyncio.Future[str | None]":
        missing: Final[str | None] = None
        return _settled(missing)


def _settled(value: _T) -> "asyncio.Future[_T]":
    future: Final = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


def _pending() -> "asyncio.Future[bool]":
    return asyncio.get_running_loop().create_future()


def _request(request_id: str, messages: object, **fields: object) -> dict[str, object]:
    return {
        "response_cost": 0.42,
        "litellm_params": {"metadata": {"litellm_lens_content": "caller value", "kept": "yes"}},
        "standard_logging_object": {
            "id": request_id,
            "litellm_call_id": f"call-{request_id}",
            "status": "success",
            "response_cost": 0.42,
            "startTime": 1_700_000_000.0,
            "metadata": {"user_api_key_hash": "hash-a", "user_api_key_team_id": "team-a"},
            "messages": messages,
            "response": {"choices": [{"message": {"content": "done"}}]},
            **fields,
        },
    }


class _Lens:
    def __init__(self) -> None:
        self.rows: Final = asyncio.Queue[dict[str, JsonValue]]()
        self.bodies: Final = asyncio.Queue[bytes]()
        self.arrived: Final = asyncio.Event()
        self.expected = 1

    def accept(self, request: httpx.Request) -> httpx.Response:
        self.bodies.put_nowait(request.content)
        for row in _ROWS.validate_json(request.content):
            self.rows.put_nowait(row)
        if self.rows.qsize() >= self.expected:
            self.arrived.set()
        return httpx.Response(204)

    def drained(self) -> list[dict[str, JsonValue]]:
        return [self.rows.get_nowait() for _ in range(self.rows.qsize())]


def _request_id(kwargs: Mapping[str, object], response: object) -> str | None:
    event: Final = as_str_mapping(kwargs.get("standard_logging_object")) or {}
    return str(event["id"]) if "id" in event else None


def _ledger_backed(
    client: httpx.AsyncClient,
    store: "_Store | None" = None,
    limits: ContentLimits = ContentLimits(),
    upload_timeout: float = 30.0,
) -> LensExporter:
    return LensExporter(
        client,
        content_store=store,
        signing_key=_KEY,
        limits=limits,
        upload_timeout=upload_timeout,
        ledger=LedgerHandoff(),
        ledger_key=_request_id,
    )


async def _log_and_commit(exporter: LensExporter, request: Mapping[str, object]) -> None:
    """Logs a request, then reports the accepted ledger delivery that confirms it with the fixture's accounting."""
    await exporter.async_log_success_event(request, None, None, None)
    request_id: Final = _request_id(request, None)
    assert request_id is not None
    exporter.ledger_committed(
        (
            {
                "request_id": request_id,
                "delivery_id": f"delivery-{request_id}",
                "spend": 0.42,
                "api_key": "hash-a",
                "team_id": "team-a",
            },
        )
    )


async def _export(
    requests: tuple[dict[str, object], ...],
    store: _Store | None = None,
    limits: ContentLimits = ContentLimits(),
) -> tuple[LensExporter, list[dict[str, JsonValue]], _Lens]:
    lens: Final = _Lens()
    lens.expected = len(requests)
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, store, limits)
        exporter.start()
        try:
            for request in requests:
                await _log_and_commit(exporter, request)
            await asyncio.wait_for(lens.arrived.wait(), timeout=5)
        finally:
            await exporter.aclose()
    return exporter, lens.drained(), lens


def _reference(row: Mapping[str, JsonValue]) -> JsonValue:
    return _OBJECT.validate_json(str(row["metadata"])).get("litellm_lens_content")


def _unavailable(reason: Unavailable) -> dict[str, str | int]:
    return dict(signed_reference(_KEY, _OWNER, reason))


def _old_record(request: dict[str, object]) -> bytes:
    payload: Final = TypeAdapter(SpendLogPayload).validate_python(request["standard_logging_object"])
    record: Final = encode_record(spend_log_row_from_payload(payload, request))
    assert isinstance(record, bytes)
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    (
        pytest.param([{"role": "user", "content": "short \u754c"}], id="small"),
        pytest.param([{"role": "user", "content": "x" * (MAX_EVENT_BYTES - 20_000)}], id="near-limit"),
    ),
)
async def test_inline_records_are_byte_identical_and_ignore_unexported_fields(messages: object) -> None:
    request: Final = _request("request-a", messages, model_map_information={"notes": "y" * 64_000})
    original: Final = copy.deepcopy(request)
    exporter, rows, lens = await _export((request,), _Store())
    assert request == original
    assert lens.bodies.get_nowait() == b"[" + _old_record(request) + b"]"
    assert _OBJECT.validate_json(str(rows[0]["metadata"]))["litellm_lens_content"] == "caller value"
    assert (exporter.rows_written, exporter.rows_dropped) == (1, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    (
        pytest.param([{"role": "user", "content": "x" * MAX_EVENT_BYTES}], id="ascii"),
        pytest.param([{"role": "user", "content": "\u754c" * 200_000}], id="escaped-unicode"),
    ),
)
async def test_oversized_content_without_storage_keeps_cost_with_a_signed_unavailable_state(messages: object) -> None:
    exporter, rows, _ = await _export((_request("request-a", messages),))
    assert (exporter.rows_written, exporter.rows_dropped) == (1, 0)
    assert (rows[0]["spend"], rows[0]["messages"], rows[0]["response"]) == (0.42, "", "")
    assert _reference(rows[0]) == _unavailable(Unavailable.STORAGE_NOT_CONFIGURED)
    assert _OBJECT.validate_json(str(rows[0]["metadata"]))["kept"] == "yes"


@pytest.mark.asyncio
async def test_content_is_referenced_only_after_a_confirmed_upload_bound_to_its_owner() -> None:
    gate: Final = _pending()
    store: Final = _Store(lambda: gate)
    messages: Final = [{"role": "user", "content": "\u754c" * 3000}]
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, store, _SMALL_INLINE)
        exporter.start()
        try:
            await _log_and_commit(exporter, _request("request-a", messages))
            await asyncio.wait_for(store.started.wait(), timeout=5)
            assert (exporter.buffered_events, lens.rows.qsize(), exporter.native_uploads) == (0, 0, 1)
            gate.set_result(True)
            await asyncio.wait_for(lens.arrived.wait(), timeout=5)
        finally:
            if not gate.done():
                gate.set_result(False)
            await exporter.aclose()
    key, body = store.uploads.get_nowait()
    row: Final = lens.drained()[0]
    reference: Final = StoredReference.model_validate(_reference(row))
    stored: Final = verified_stored(_KEY, _OWNER, reference)
    assert stored is not None and stored.object_key == key and key.startswith("logs/lens-content/2023-11-14/")
    assert (stored.size, stored.sha256) == (len(body.encode()), hashlib.sha256(body.encode()).hexdigest())
    assert verified_stored(_KEY, Owner(team_id="team-b", api_key="hash-a", request_id="request-a"), reference) is None
    assert _TEXT.validate_json(body)["messages"] == json.dumps(messages)
    assert (row["messages"], row["response"], row["spend"]) == ("", "", 0.42)
    assert (exporter.native_uploads, exporter.pending_content_bytes) == (0, 0)


async def _rejected() -> bool:
    return False


async def _raises() -> bool:
    raise RuntimeError("private storage host")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", (_rejected, _raises), ids=("not-delivered", "error"))
async def test_failed_upload_keeps_cost_and_reports_upload_failed(
    outcome: Callable[[], Coroutine[object, object, bool]],
) -> None:
    store: Final = _Store(lambda: asyncio.ensure_future(outcome()))
    exporter, rows, _ = await _export((_request("request-a", "x" * 100_000),), store, _SMALL_INLINE)
    assert store.uploads.qsize() == 1
    assert _reference(rows[0]) == _unavailable(Unavailable.UPLOAD_FAILED)
    assert (rows[0]["spend"], exporter.rows_dropped) == (0.42, 0)


def _settle_pending(store: _Store) -> None:
    for native in tuple(store.natives.get_nowait() for _ in range(store.natives.qsize())):
        if not native.done():
            native.set_result(False)


async def _reservation_released(native: "asyncio.Future[bool]", result: bool) -> None:
    released: Final = asyncio.Event()
    native.add_done_callback(lambda _: released.set())
    native.set_result(result)
    await asyncio.wait_for(released.wait(), timeout=5)


@pytest.mark.asyncio
async def test_upload_timeout_exports_the_record_but_holds_native_capacity_until_completion() -> None:
    store: Final = _Store(_pending)
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, store, _SMALL_INLINE, upload_timeout=0)
        exporter.start()
        try:
            await _log_and_commit(exporter, _request("request-a", "x" * 100_000))
            await asyncio.wait_for(lens.arrived.wait(), timeout=5)
            assert _reference(lens.drained()[0]) == _unavailable(Unavailable.UPLOAD_FAILED)
            assert exporter.native_uploads == 1 and exporter.pending_content_bytes > 0
            await _reservation_released(store.natives.get_nowait(), True)
            assert (exporter.native_uploads, exporter.pending_content_bytes) == (0, 0)
        finally:
            _settle_pending(store)
            await exporter.aclose()


@pytest.mark.asyncio
async def test_shutdown_exports_cancelled_records_while_native_uploads_keep_their_reservation() -> None:
    store: Final = _Store(_pending)
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, store, _SMALL_INLINE)
        exporter.start()
        try:
            await _log_and_commit(exporter, _request("request-a", "x" * 100_000))
            await asyncio.wait_for(store.started.wait(), timeout=5)
        finally:
            await exporter.aclose()
        try:
            assert _reference(lens.drained()[0]) == _unavailable(Unavailable.CANCELLED)
            assert exporter.native_uploads == 1
            await _reservation_released(store.natives.get_nowait(), True)
        finally:
            _settle_pending(store)
    assert (exporter.native_uploads, exporter.pending_content_bytes) == (0, 0)


@pytest.mark.asyncio
async def test_native_upload_saturation_rejects_new_content_before_serializing_it() -> None:
    store: Final = _Store(_pending)
    lens: Final = _Lens()
    lens.expected = MAX_NATIVE_UPLOADS + 2
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, store, _SMALL_INLINE)
        exporter.start()
        try:
            for index in range(MAX_NATIVE_UPLOADS):
                await _log_and_commit(exporter, _request(f"r{index}", "x" * 100_000))
            while store.natives.qsize() < MAX_NATIVE_UPLOADS:
                store.started.clear()
                await asyncio.wait_for(store.started.wait(), timeout=5)
            await _log_and_commit(exporter, _request("overflow", "x" * 100_000))
            await _log_and_commit(exporter, _request("overflow-after-commit", "x" * 5000))
            assert store.uploads.qsize() == MAX_NATIVE_UPLOADS
            for _ in range(MAX_NATIVE_UPLOADS):
                await _reservation_released(store.natives.get_nowait(), True)
            await asyncio.wait_for(lens.arrived.wait(), timeout=5)
        finally:
            _settle_pending(store)
            await exporter.aclose()
    states: Final = {row["request_id"]: _reference(row) for row in lens.drained()}
    overflow: Final = Owner(team_id="team-a", api_key="hash-a", request_id="overflow")
    assert states["overflow"] == dict(signed_reference(_KEY, overflow, Unavailable.OVERLOADED))
    late: Final = Owner(team_id="team-a", api_key="hash-a", request_id="overflow-after-commit")
    assert states["overflow-after-commit"] == dict(signed_reference(_KEY, late, Unavailable.OVERLOADED))
    assert sum(isinstance(state, dict) and state["state"] == "stored" for state in states.values()) == (
        MAX_NATIVE_UPLOADS
    )


class _Unprintable:
    def __str__(self) -> str:
        raise TypeError("cannot be rendered")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "limits", "reason"),
    (
        pytest.param("x" * 9000, ContentLimits(inline_bytes=4096, object_bytes=8192), "too_large", id="object-limit"),
        pytest.param(
            [{"content": "\u754c" * 1500}],
            ContentLimits(inline_bytes=4096, object_bytes=8192),
            "too_large",
            id="unicode-escape-expansion",
        ),
        pytest.param("\ud800" * 100_000, _SMALL_INLINE, "unserializable", id="lone-surrogate"),
        pytest.param([_Unprintable()], _SMALL_INLINE, "unserializable", id="encoder-failure"),
        pytest.param(
            "x" * 8180, ContentLimits(inline_bytes=4096, object_bytes=8192), "too_large", id="response-budget"
        ),
        pytest.param('"' * 5000, ContentLimits(inline_bytes=4096, object_bytes=8192), "too_large", id="object-escape"),
    ),
)
async def test_unstorable_content_is_never_uploaded_and_keeps_cost(
    messages: object, limits: ContentLimits, reason: str
) -> None:
    store: Final = _Store()
    _, rows, _ = await _export((_request("request-a", messages),), store, limits)
    assert store.uploads.empty()
    assert _reference(rows[0]) == _unavailable(Unavailable(reason))
    assert rows[0]["spend"] == 0.42


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store", "fields", "reason"),
    (
        pytest.param(lambda: None, {}, "storage_not_configured", id="fits-as-text-not-as-row"),
        pytest.param(lambda: _Store(prefix=None), {}, "storage_not_configured", id="no-object-key"),
        pytest.param(_Store, {"startTime": 1e20}, "unserializable", id="no-object-name"),
    ),
)
async def test_content_that_cannot_be_named_or_stored_keeps_its_committed_cost(
    store: Callable[[], "_Store | None"], fields: dict[str, object], reason: str
) -> None:
    _, rows, _ = await _export((_request("request-a", "x" * 4000, **fields),), store(), _SMALL_INLINE)
    assert _reference(rows[0]) == _unavailable(Unavailable(reason))
    assert (rows[0]["spend"], rows[0]["messages"]) == (0.42, "")


@pytest.mark.asyncio
async def test_a_serializer_that_cannot_start_releases_its_slot_and_keeps_the_committed_cost() -> None:
    await asyncio.get_running_loop().shutdown_default_executor()
    store: Final = _Store()
    exporter, rows, _ = await _export((_request("request-a", "x" * 100_000),), store, _SMALL_INLINE)
    assert _reference(rows[0]) == _unavailable(Unavailable.UPLOAD_FAILED)
    assert (rows[0]["spend"], store.uploads.empty(), exporter.serialize_slots.locked()) == (0.42, True, False)
    assert exporter.pending_content_bytes == 0


@pytest.mark.asyncio
async def test_configured_inline_threshold_moves_smaller_content_to_storage() -> None:
    store: Final = _Store()
    _, rows, _ = await _export((_request("request-a", "x" * 5000),), store, _SMALL_INLINE)
    assert store.uploads.qsize() == 1
    assert isinstance(reference := _reference(rows[0]), dict) and reference["state"] == "stored"


@pytest.mark.asyncio
async def test_redacted_content_is_never_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    store: Final = _Store()
    _, rows, _ = await _export((_request("request-a", "x" * MAX_EVENT_BYTES),), store)
    assert store.uploads.empty()
    assert (rows[0]["messages"], rows[0]["response"], rows[0]["spend"]) == ("", "", 0.42)
    assert _reference(rows[0]) is None


def test_no_log_requests_never_reach_the_lens_exporter() -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = Logging(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime(2023, 11, 14, tzinfo=timezone.utc),
        litellm_call_id="call-no-log",
        function_id="function-no-log",
    )
    exporter: Final = LensExporter(httpx.AsyncClient(base_url="http://lens"))
    assert logging_obj.should_run_callback(exporter, {"no-log": True}, "async_success") is False
    assert logging_obj.should_run_callback(exporter, {}, "async_success") is True


class _FailingStore(_Store):
    def upload(self, key: str, body: str) -> "asyncio.Future[bool]":
        raise RuntimeError("the native sink rejected the call")


@pytest.mark.asyncio
async def test_a_store_that_cannot_start_an_upload_releases_its_reservation() -> None:
    exporter, rows, _ = await _export((_request("request-a", "x" * 100_000),), _FailingStore(), _SMALL_INLINE)
    assert _reference(rows[0]) == _unavailable(Unavailable.UPLOAD_FAILED)
    assert (exporter.native_uploads, exporter.pending_content_bytes, rows[0]["spend"]) == (0, 0, 0.42)


class _BlockingValue:
    def __init__(self) -> None:
        self.entered: Final = threading.Event()
        self.release: Final = threading.Event()

    def __str__(self) -> str:
        self.entered.set()
        self.release.wait()
        return "released"


@pytest.mark.asyncio
async def test_raw_content_stays_reserved_until_its_serialization_thread_finishes() -> None:
    from litellm.integrations.lens.content import MAX_SERIALIZATIONS

    blocker: Final = _BlockingValue()
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, _Store(), _SMALL_INLINE)
        exporter.start()
        try:
            await _log_and_commit(exporter, _request("request-a", ["x" * 70_000, blocker]))
            assert exporter.pending_content_bytes >= 70_000
            assert await asyncio.to_thread(blocker.entered.wait, 5)
            await exporter.aclose()
            assert _reference(lens.drained()[0]) == _unavailable(Unavailable.CANCELLED)
            assert exporter.pending_content_bytes >= 70_000
        finally:
            blocker.release.set()
            await exporter.aclose()
        for _ in range(MAX_SERIALIZATIONS):
            await asyncio.wait_for(exporter.serialize_slots.acquire(), timeout=5)
        assert exporter.pending_content_bytes == 0


@pytest.mark.asyncio
async def test_closing_before_a_content_job_starts_exports_its_cost_and_releases_reservations() -> None:
    store: Final = _Store()
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_backed(client, store, _SMALL_INLINE)
        exporter.start()
        try:
            await _log_and_commit(exporter, _request("request-a", "x" * 100_000))
            assert exporter.pending_content_bytes >= 100_000
        finally:
            await exporter.aclose()
    rows: Final = lens.drained()
    assert len(rows) == 1
    assert _reference(rows[0]) == _unavailable(Unavailable.CANCELLED)
    assert rows[0]["spend"] == 0.42
    assert store.uploads.empty()
    assert (exporter.pending_content_bytes, len(exporter.content_tasks), exporter.serialize_slots.locked()) == (
        0,
        0,
        False,
    )


_CANONICAL: Final = Owner(team_id="team-canonical", api_key="hash-canonical", request_id="request-a")


def _committed(spend: float = 0.75) -> dict[str, object]:
    return {
        "request_id": "request-a",
        "delivery_id": "delivery-a",
        "spend": spend,
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "api_key": _CANONICAL.api_key,
        "team_id": _CANONICAL.team_id,
    }


def _ledger_exporter(client: httpx.AsyncClient, store: _Store, key: str | None = "request-a") -> LensExporter:
    from litellm.integrations.lens.ledger import LedgerHandoff

    return LensExporter(
        client,
        content_store=store,
        signing_key=_KEY,
        limits=_SMALL_INLINE,
        ledger=LedgerHandoff(),
        ledger_key=lambda kwargs, response: key,
    )


@pytest.mark.asyncio
async def test_oversized_ledger_content_is_stored_only_after_commit_under_the_committed_owner() -> None:
    store: Final = _Store()
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_exporter(client, store)
        exporter.start()
        try:
            await exporter.async_log_success_event(_request("request-a", "x" * 5000), None, None, None)
            assert (store.uploads.empty(), exporter.buffered_events) == (True, 0)
            exporter.ledger_committed((_committed(),))
            await asyncio.wait_for(lens.arrived.wait(), timeout=5)
        finally:
            _settle_pending(store)
            await exporter.aclose()
    key, body = store.uploads.get_nowait()
    row: Final = lens.drained()[0]
    reference: Final = StoredReference.model_validate(_reference(row))
    callback_owner: Final = Owner(team_id="team-a", api_key="hash-a", request_id="request-a")
    assert verified_stored(_KEY, _CANONICAL, reference) is not None
    assert verified_stored(_KEY, callback_owner, reference) is None
    assert reference.sha256 == hashlib.sha256(body.encode()).hexdigest() and key == reference.object_key
    assert (row["team_id"], row["api_key"], row["spend"], row["total_tokens"]) == (
        _CANONICAL.team_id,
        _CANONICAL.api_key,
        0.75,
        5,
    )
    assert (row["messages"], row["response"]) == ("", "")


@pytest.mark.asyncio
async def test_an_uncommitted_ledger_record_is_never_uploaded_or_exported() -> None:
    store: Final = _Store()
    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporter: Final = _ledger_exporter(client, store)
        try:
            await exporter.async_log_success_event(_request("request-a", "x" * 5000), None, None, None)
        finally:
            await exporter.aclose()
    assert store.uploads.empty()
    assert (exporter.rows_written, exporter.rows_dropped, exporter.pending_content_bytes) == (0, 1, 0)


@pytest.mark.asyncio
async def test_content_without_a_ledger_commit_to_wait_for_is_never_stored_or_sent() -> None:
    store: Final = _Store()
    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporter: Final = _ledger_exporter(client, store, key=None)
        try:
            await exporter.async_log_success_event(_request("request-a", "x" * 100_000), None, None, None)
            exporter.ledger_committed((_committed(),))
        finally:
            await exporter.aclose()
    assert store.uploads.empty()
    assert (exporter.rows_written, exporter.rows_dropped, exporter.pending_content_bytes) == (0, 2, 0)


@pytest.mark.asyncio
async def test_a_failed_upload_after_commit_keeps_the_committed_cost() -> None:
    store: Final = _Store(lambda: asyncio.ensure_future(_rejected()))
    lens: Final = _Lens()
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lens.accept)) as client:
        exporter: Final = _ledger_exporter(client, store)
        exporter.start()
        try:
            await exporter.async_log_success_event(_request("request-a", "x" * 5000), None, None, None)
            exporter.ledger_committed((_committed(spend=1.25),))
            await asyncio.wait_for(lens.arrived.wait(), timeout=5)
        finally:
            await exporter.aclose()
    row: Final = lens.drained()[0]
    assert _reference(row) == dict(signed_reference(_KEY, _CANONICAL, Unavailable.UPLOAD_FAILED))
    assert (row["spend"], row["team_id"], store.uploads.qsize()) == (1.25, _CANONICAL.team_id, 1)


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


def _metadata_only(request_id: str, padding: str) -> dict[str, object]:
    return {
        "response_cost": 9.0,
        "litellm_call_id": request_id,
        "standard_logging_object": {
            "id": request_id,
            "status": "success",
            "response_cost": 9.0,
            "metadata": {"padding": padding},
        },
    }


@pytest.mark.asyncio
async def test_held_records_are_charged_their_retained_metadata_so_the_handoff_budget_evicts_them() -> None:
    sent: Final = asyncio.Queue[bytes]()

    def accept(request: httpx.Request) -> httpx.Response:
        sent.put_nowait(request.content)
        return httpx.Response(204)

    handoff: Final = LedgerHandoff(max_bytes=50_000)
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(
            client, ledger=handoff, ledger_key=lambda kwargs, response: str(kwargs.get("litellm_call_id"))
        )
        exporter.start()
        try:
            for request_id in ("first", "second"):
                await exporter.async_log_success_event(_metadata_only(request_id, "界" * 5000), None, None, None)
            assert (tuple(handoff.held), exporter.rows_dropped, sent.empty()) == (("second",), 1, True)
            assert 30_000 < handoff.held_bytes <= handoff.max_bytes
            exporter.ledger_committed(({"request_id": "second", "delivery_id": "d-second", "spend": 0.25},))
            body: Final = await asyncio.wait_for(sent.get(), timeout=5)
        finally:
            await exporter.aclose()
    rows: Final = _ROWS.validate_json(body)
    assert [(row["request_id"], row["spend"]) for row in rows] == [("second", 0.25)]
    assert (exporter.rows_written, exporter.rows_dropped, handoff.held_bytes) == (1, 1, 0)
