import asyncio
import json
import logging
from datetime import datetime, timezone
from functools import partial
from itertools import chain
from typing import Final

import httpx
import prometheus_client
import pytest
from fastapi import Request
from prometheus_client import CollectorRegistry, Counter

from litellm.integrations.lens import exporter as exporter_module
from litellm.integrations.lens.exporter import (
    MAX_BUFFER_EVENTS,
    MAX_EVENT_BYTES,
    ExportFailure,
    LensExporter,
    encode_record,
    lens_export_counters,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import HttpPassThroughEndpointHelpers
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True), ids=("success", "failure"))
@pytest.mark.parametrize(
    ("key_value", "team_value", "expected_count"),
    ((None, None, 1), (None, False, 0), (False, True, 0), (True, False, 1)),
    ids=("configured-default", "team-off", "key-off", "key-overrides-team"),
)
async def test_passthrough_request_records_obey_the_authenticated_lens_policy(
    failed: bool, key_value: bool | None, team_value: bool | None, expected_count: int
) -> None:
    at: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    auth: Final = UserAPIKeyAuth(metadata={"lens_enabled": key_value}, team_metadata={"lens_enabled": team_value})
    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/gemini/v1beta/models/test-model:generateContent",
            "query_string": b"",
            "headers": [],
            "server": ("gateway.test", 80),
        }
    )
    logging_obj: Final = Logging(
        model="test-model",
        messages=[],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=at,
        litellm_call_id="passthrough-policy",
        function_id="passthrough-policy",
    )
    kwargs: Final = HttpPassThroughEndpointHelpers.init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=auth,
        passthrough_logging_payload={},
        logging_obj=logging_obj,
        _parsed_body={
            "metadata": {
                "lens_enabled": True,
                "user_api_key_auth": {"metadata": {"lens_enabled": True}},
            }
        },
        litellm_call_id="passthrough-policy",
    )
    received: Final = asyncio.Queue[httpx.Request]()

    def accept(export_request: httpx.Request) -> httpx.Response:
        received.put_nowait(export_request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client)
        exporter.start()
        callback: Final = exporter.async_log_failure_event if failed else exporter.async_log_success_event
        await callback(
            {
                **kwargs,
                "standard_logging_object": {
                    "id": "passthrough-policy",
                    "call_type": "pass_through_endpoint",
                    "status": "failure" if failed else "success",
                    "messages": [{"role": "user", "content": "passthrough request content"}],
                    "response": {"text": "passthrough response content"},
                },
            },
            None,
            at,
            at,
        )
        await exporter.aclose()

    assert received.qsize() == expected_count
    assert exporter.rows_written == expected_count
    assert exporter.rows_dropped == 0
    assert "user_api_key_auth" not in kwargs["litellm_params"]["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True), ids=("success", "failure"))
@pytest.mark.parametrize(
    ("key_metadata", "team_metadata", "expected_count"),
    (
        pytest.param(None, None, 1, id="global-default"),
        pytest.param({}, {"lens_enabled": False}, 0, id="team-disabled"),
        pytest.param({"lens_enabled": None}, {"lens_enabled": False}, 0, id="key-inherits-disabled-team"),
        pytest.param({"lens_enabled": False}, {"lens_enabled": True}, 0, id="key-disabled"),
        pytest.param({"lens_enabled": True}, {"lens_enabled": False}, 1, id="key-overrides-disabled-team"),
    ),
)
async def test_lens_policy_controls_success_and_failure_exports(
    failed: bool,
    key_metadata: dict[str, object] | None,
    team_metadata: dict[str, object] | None,
    expected_count: int,
) -> None:
    requests: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(204)

    auth: Final = UserAPIKeyAuth(metadata=key_metadata or {}, team_metadata=team_metadata)
    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client)
        exporter.start()
        callback: Final = exporter.async_log_failure_event if failed else exporter.async_log_success_event
        await callback(
            {
                "litellm_params": {
                    "metadata": {
                        "user_api_key_auth": auth,
                        "lens_enabled": True,
                        "user_api_key_metadata": {"lens_enabled": True},
                    }
                },
                "standard_logging_object": {"id": "policy-request", "status": "failure" if failed else "success"},
            },
            None,
            None,
            None,
        )
        await exporter.aclose()
    assert requests.qsize() == expected_count
    assert exporter.rows_written == expected_count
    assert exporter.rows_dropped == 0


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
        exporter: Final = LensExporter(client)
        exporter.start()
        callback: Final = exporter.async_log_failure_event if failed else exporter.async_log_success_event
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
    assert (rows[0]["messages"], rows[0]["response"]) == ("", "")
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
        {"id": "x", "response_cost": float("nan")},
    ),
    ids=("missing", "invalid-id", "invalid-row-number"),
)
async def test_invalid_callback_data_never_interrupts_model_requests(payload: object) -> None:
    async with httpx.AsyncClient(base_url="http://lens") as client:
        exporter: Final = LensExporter(client)
        await exporter.async_log_failure_event({"standard_logging_object": payload}, None, None, None)
        await exporter.aclose()
    assert exporter.rows_written == 0
    assert exporter.buffered_bytes == 0
    assert exporter.rows_dropped == (0 if payload is None else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation",
    (
        {"messages": [{"role": "user", "content": "x" * MAX_EVENT_BYTES}], "response": {"id": "small"}},
        {"messages": [{"role": "user", "content": "\x01" * (MAX_EVENT_BYTES // 4)}], "response": {"id": "small"}},
        {"messages": [{"role": "user", "content": "hi"}], "response": {"text": "x" * MAX_EVENT_BYTES}},
        {"model_parameters": {"tools": [{"type": "function", "strict": False}] * (MAX_EVENT_BYTES // 64)}},
    ),
    ids=("long-conversation", "grows-past-the-budget-when-encoded", "long-response", "many-tool-definitions"),
)
async def test_a_request_with_unexported_content_over_the_budget_reaches_lens_with_its_cost(
    conversation: dict[str, object],
) -> None:
    received: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        received.put_nowait(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens.test", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client)
        exporter.start()
        await exporter.async_log_success_event(
            {
                "response_cost": 0.42,
                "standard_logging_object": {
                    "id": "long-1",
                    "status": "success",
                    "response_cost": 0.42,
                    "prompt_tokens": 250000,
                    "metadata": {"user_api_key_hash": "hash-test"},
                    **conversation,
                },
            },
            None,
            None,
            None,
        )
        await exporter.aclose()
    requests: Final = tuple(received.get_nowait() for _ in range(received.qsize()))
    rows: Final = tuple(chain.from_iterable(json.loads(request.content) for request in requests))
    assert [(row["request_id"], row["spend"], row["prompt_tokens"], row["api_key"]) for row in rows] == [
        ("long-1", 0.42, 250000, "hash-test")
    ]
    assert (rows[0]["messages"], rows[0]["response"]) == ("", "")
    assert exporter.rows_dropped == 0


@pytest.mark.asyncio
async def test_a_request_too_large_without_its_conversation_is_dropped() -> None:
    sent: Final = asyncio.Queue[httpx.Request]()

    def accept(request: httpx.Request) -> httpx.Response:
        sent.put_nowait(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(base_url="http://lens.test", transport=httpx.MockTransport(accept)) as client:
        exporter: Final = LensExporter(client)
        exporter.start()
        await exporter.async_log_success_event(
            {
                "standard_logging_object": {
                    "id": "huge-metadata",
                    "status": "success",
                    "metadata": {"note": "x" * MAX_EVENT_BYTES},
                    "messages": [{"role": "user", "content": "hi"}],
                }
            },
            None,
            None,
            None,
        )
        await exporter.aclose()
    assert sent.empty()
    assert exporter.rows_dropped == 1
    assert exporter.rows_written == 0


def test_serialization_rejects_recursive_payloads() -> None:
    cyclic: Final[dict[str, object]] = {}  # mutable-ok: deliberately constructs a cyclic callback payload
    cyclic["self"] = cyclic
    assert encode_record(cyclic) is ExportFailure.TOO_LARGE


@pytest.mark.parametrize("value", (float("nan"), object(), "\ud800"), ids=("nan", "unsupported", "surrogate"))
def test_serialization_returns_a_failure_for_invalid_payloads(value: object) -> None:
    assert encode_record({"value": value}) is ExportFailure.INVALID


async def _drained(exporter: LensExporter) -> None:
    while exporter.buffered_events:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_written_and_dropped_rows_are_exported_as_prometheus_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry: Final = CollectorRegistry()
    # Exercise the real factory without sharing its cached collectors with other tests.
    monkeypatch.setattr(prometheus_client, "Counter", partial(Counter, registry=registry))
    monkeypatch.setattr(exporter_module, "lens_export_counters", lens_export_counters.__wrapped__)
    statuses: Final = iter((204, 500))

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses))

    async with httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(answer)) as client:
        exporter: Final = LensExporter(client)
        exporter.start()
        for request_id in ("accepted", "rejected"):
            await exporter.async_log_success_event(
                {"standard_logging_object": {"id": request_id, "response_cost": 0.1}}, None, None, None
            )
            await _drained(exporter)
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
