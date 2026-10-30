import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from litellm.accounting import Metric, MetricName, Outcome
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.db.accounting_ledger import AccountingConflictError, UsageWrite
from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureError, UsageCaptureSettings
from litellm.types.utils import CallTypes
from tests.unit.proxy.spend_tracking.test_usage_meter import CommitBarrier, _manual_flush, _write


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}])
async def test_capture_keeps_unknown_usage_distinct_from_measured_zero(usage: Mapping[str, int] | None) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = CommitBarrier()
    database.release.set()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    captured: Final = capture_callback(
        meter,
        {"call_type": "acompletion", "litellm_call_id": "caller", "model": "unknown-model", "litellm_params": {}},
        {"id": "provider", "usage": usage},
        timestamp,
        timestamp,
    )
    await meter.close()
    outbound: Final = captured.writes[1].event
    assert outbound.metrics == (
        None
        if usage is None
        else tuple(Metric(name=MetricName(key), value=Decimal(value)) for key, value in usage.items())
    )
    envelope: Final = json.loads(captured.writes[1].envelope)
    assert envelope["usage"] == usage
    assert envelope["pricing"]["reported_cost"] is None
    assert database.committed == ("caller:inbound", "provider")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    [
        call.value
        for call in (
            CallTypes.acompletion,
            CallTypes.arealtime,
            CallTypes.call_mcp_tool,
            CallTypes.asend_message,
            CallTypes.pass_through,
        )
    ],
)
async def test_shared_callback_capture_retains_usage_and_attribution_without_bodies(call_type: str) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = CommitBarrier()
    database.release.set()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    captured: Final = capture_callback(
        meter,
        {
            "call_type": call_type,
            "litellm_call_id": "caller",
            "model": "test-model",
            "messages": "private prompt",
            "litellm_params": {"api_key": "private-secret", "metadata": {"user_api_key_team_id": "team"}},
        },
        {
            "id": "provider",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "output": "private answer",
        },
        timestamp,
        timestamp,
    )
    await meter.close()
    envelope: Final = captured.writes[1].envelope
    assert all(secret not in envelope for secret in ("private prompt", "private-secret", "private answer"))
    event: Final = captured.writes[1].event
    assert event.call_type.value == call_type
    assert event.metrics == (
        Metric(name=MetricName.PROMPT_TOKENS, value=Decimal(3)),
        Metric(name=MetricName.COMPLETION_TOKENS, value=Decimal(2)),
        Metric(name=MetricName.TOTAL_TOKENS, value=Decimal(5)),
    )
    assert any(dimension.value == "team" for dimension in event.dimensions)


@pytest.mark.asyncio
async def test_cache_duplicate_uses_caller_identity_despite_timestamped_diagnostic_ids() -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = CommitBarrier()
    database.release.set()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    kwargs: Final = {
        "call_type": "acompletion",
        "litellm_call_id": "caller",
        "model": "test-model",
        "cache_hit": True,
        "litellm_params": {},
    }
    response: Final = {"id": "cached-provider-id", "usage": {"total_tokens": 5}}
    first: Final = capture_callback(meter, kwargs, response, timestamp, timestamp)
    second: Final = capture_callback(meter, kwargs, response, timestamp, timestamp)
    await meter.close()
    assert first.writes == second.writes
    assert first.writes[1].event.event_id == "caller_cache_hit"
    assert json.loads(first.writes[1].envelope)["charge_eligible"] is False


NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class StoredUsage:
    def __init__(self) -> None:
        self.rows: tuple[UsageWrite, ...] = ()

    async def load(self, identities: tuple[str, ...]) -> tuple[UsageWrite, ...]:
        return tuple(row for row in self.rows if row.event.event_id in identities)

    async def insert(self, writes: tuple[UsageWrite, ...]) -> None:
        existing: Final = {row.event.event_id: row for row in self.rows}
        if any(existing.get(row.event.event_id, row) != row for row in writes):
            raise AccountingConflictError("Same identity has different content")
        self.rows += tuple(row for row in writes if row.event.event_id not in existing)


def _logger(call_type: str = "acompletion", *, stream: bool = False) -> Logging:
    from litellm.litellm_core_utils.litellm_logging import Logging

    logger: Final = Logging(
        model="capture-model",
        messages=[],
        stream=stream,
        call_type=call_type,
        start_time=NOW,
        litellm_call_id="caller",
        function_id="function",
    )
    logger.update_environment_variables(
        model="capture-model", custom_llm_provider="openai", litellm_params={}, optional_params={}
    )
    return logger


@pytest.mark.asyncio
async def test_terminal_retrieval_reuses_accepted_snapshot_but_rejects_changed_usage() -> None:
    from datetime import timedelta

    from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
    from litellm.proxy.spend_tracking.usage_capture import UsageBatchWriter, capture_callback
    from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

    database: Final = StoredUsage()
    writer: Final = UsageBatchWriter(database.insert, database.load)
    meter: Final = DurableUsageMeter(writer.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    kwargs: Final = {
        "call_type": "aget_responses",
        "litellm_call_id": "poll-one",
        "model": "capture-model",
        "response_cost": 2.0,
        "litellm_params": {"metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN}},
    }
    response: Final = {
        "id": "job",
        "status": "completed",
        "background": True,
        "created_at": NOW.timestamp(),
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    }
    first: Final = capture_callback(meter, kwargs, response, NOW, NOW)
    await meter.flush()
    later: Final = capture_callback(
        meter, {**kwargs, "litellm_call_id": "poll-two", "response_cost": 9.0}, response, NOW, NOW + timedelta(hours=2)
    )
    await meter.flush()
    await later.receipt
    assert database.rows == first.writes
    assert len(database.rows) == 1
    assert database.rows[0].event.event_id == "job_background_cost"
    changed: Final = capture_callback(meter, kwargs, {**response, "usage": {"total_tokens": 8}}, NOW, NOW)
    await meter.close()
    with pytest.raises(UsageCaptureError, match="identity"):
        await changed.receipt
    assert database.rows == first.writes


@pytest.mark.asyncio
async def test_batch_poll_has_distinct_identity_and_only_terminal_usage_is_chargeable() -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    kwargs: Final = {
        "call_type": "aretrieve_batch",
        "litellm_call_id": "poll",
        "model": "capture-model",
        "litellm_params": {},
    }
    pending: Final = capture_callback(meter, kwargs, {"id": "batch", "status": "in_progress"}, NOW, NOW)
    completed: Final = capture_callback(
        meter, kwargs, {"id": "batch", "status": "completed", "usage": {"total_tokens": 9}}, NOW, NOW
    )
    await meter.close()
    assert pending.writes[-1].event.event_id == "poll:status"
    assert json.loads(pending.writes[-1].envelope)["charge_eligible"] is False
    assert completed.writes[-1].event.event_id == "batch_batch_cost"
    assert json.loads(completed.writes[-1].envelope)["charge_eligible"] is True
    assert len(completed.writes) == 1


@pytest.mark.asyncio
async def test_internal_capture_does_not_claim_reused_parent_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm
    from litellm._internal_context import internal_sub_call
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setitem(
        litellm.model_cost,
        "capture-model",
        {
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.2,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger()
    response: Final = ModelResponse(
        id="response", model="capture-model", usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
    )
    with internal_sub_call("caller"):
        capture_logging_usage(meter, logger, response, NOW, NOW)
    monkeypatch.setitem(
        litellm.model_cost,
        "capture-model",
        {
            "input_cost_per_token": 0.7,
            "output_cost_per_token": 0.9,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    litellm.register_model({"capture-model": litellm.model_cost["capture-model"]})
    capture_logging_usage(meter, logger, response, NOW, NOW)
    capture_logging_usage(meter, logger, response, NOW, NOW)
    await meter.close()
    assert tuple(row.event.event_id for row in database.rows) == ("response:included", "caller:inbound", "response")
    child: Final = json.loads(database.rows[0].envelope)
    parent: Final = json.loads(database.rows[-1].envelope)
    assert (child["ownership"], child["charge_eligible"], child["parent_request_id"]) == (
        "included_in_parent",
        False,
        "caller",
    )
    assert parent["charge_eligible"] is True
    assert child["pricing"]["model_map_information"]["model_map_value"]["input_cost_per_token"] == 0.1
    assert parent["pricing"]["model_map_information"]["model_map_value"]["input_cost_per_token"] == 0.7


@pytest.mark.asyncio
async def test_stream_capture_ignores_chunks_and_preserves_terminal_usage_extras() -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger(stream=True)
    capture_logging_usage(meter, logger, ModelResponseStream(id="chunk"), NOW, NOW)
    assert meter.pending_records == 0
    capture_logging_usage(
        meter, logger, ModelResponse(id="stream", usage=Usage(prompt_tokens=3, provider_units=4)), NOW, NOW
    )
    await meter.close()
    assert json.loads(database.rows[-1].envelope)["usage"]["provider_units"] == 4
    assert database.rows[-1].event.event_id == "stream"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_policy", ["fail_closed", "log"])
async def test_sync_ack_commit_failure_follows_response_policy(
    failure_policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier(failure=ValueError("invalid envelope"))
    database.release.set()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=2))
    meter: Final = runtime.meter
    assert meter is not None
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"accounting_sync_ack": True, "accounting_sync_ack_failure_policy": failure_policy},
    )
    logger: Final = _logger()
    response: Final = ModelResponse(id="provider", usage=Usage(prompt_tokens=4))
    if failure_policy == "fail_closed":
        with pytest.raises(HTTPException) as error:
            await capture_proxy_response(logger, response)
        assert error.value.status_code == 503
    else:
        await capture_proxy_response(logger, response)
    await runtime.stop(1.0)
    assert database.committed == ()
    assert meter.pending_records == 0


@pytest.mark.asyncio
async def test_failed_attempt_keeps_its_usage_without_claiming_the_callers_success() -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    try:
        logger.record_partial_usage_for_failure(Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7), 0.4)
        logger.carry_failed_attempt_usage()
        logger.capture_usage_before_logging(
            ModelResponse(id="fallback", usage=Usage(prompt_tokens=3, total_tokens=3)), NOW, NOW
        )
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == ("caller-attempt-0", "caller:inbound", "fallback")
    failed: Final = json.loads(database.rows[0].envelope)
    assert failed["usage"]["total_tokens"] == 7
    assert failed["parent_request_id"] == "caller"
    assert failed["request_count_eligible"] is False
    assert failed["charge_eligible"] is True
    assert failed["event"]["retry_of_attempt_id"] is None
    assert failed["event"]["outcome"] == "failed"
    assert failed["pricing"]["reported_cost"] == 0.4
    assert database.rows[1].event.outcome is Outcome.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("estimated", [False, True])
async def test_cancelled_usage_keeps_measured_or_estimated_provenance(estimated: bool) -> None:
    from fastapi import HTTPException

    from litellm.litellm_core_utils.litellm_logging import UsageObservation
    from litellm.proxy.spend_tracking.usage_capture import capture_callback
    from litellm.types.utils import Usage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    capture_callback(
        meter,
        {
            "litellm_call_id": "cancelled",
            "call_type": "acompletion",
            "model": "capture-model",
            "litellm_params": {},
            "combined_usage_object": Usage(prompt_tokens=4, total_tokens=4),
            "_accounting_usage_observed": UsageObservation.ESTIMATED if estimated else UsageObservation.KNOWN,
        },
        HTTPException(499, "Client disconnected"),
        NOW,
        NOW,
    )
    await meter.close()
    payload: Final = json.loads(database.rows[-1].envelope)
    assert payload["event"]["outcome"] == "cancelled"
    assert payload["usage"]["total_tokens"] == 4
    assert payload["usage_source"] == ("estimated" if estimated else "measured")
    assert payload["pricing"]["reported_cost"] is None


@pytest.mark.asyncio
async def test_realtime_capture_keeps_metered_units_and_provider_usage_without_frames() -> None:
    from litellm.constants import REALTIME_METERED_USAGE_KEY
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.types.llms.xai import RealtimeMeteredUsage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger(CallTypes.arealtime.value)
    logger.model_call_details[REALTIME_METERED_USAGE_KEY] = RealtimeMeteredUsage(True, 1.5, 2.5, 3)
    capture_logging_usage(
        meter,
        logger,
        [
            {
                "type": "response.done",
                "response": {
                    "id": "turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                    "output": "private realtime transcript",
                },
            }
        ],
        NOW,
        NOW,
    )
    await meter.close()
    payload: Final = json.loads(database.rows[-1].envelope)
    assert payload["usage"]["total_tokens"] == 7
    assert payload["usage"]["metered_usage"] == {
        "push_to_talk": True,
        "audio_seconds_sent": 1.5,
        "audio_seconds_received": 2.5,
        "text_messages": 3,
    }
    assert "private realtime transcript" not in database.rows[-1].envelope


@pytest.mark.asyncio
async def test_default_http_capture_returns_before_database_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=2))
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    try:
        await capture_proxy_response(
            _logger(),
            ModelResponse(id="asynchronous", usage=Usage(prompt_tokens=1)),
        )
        await database.entered.wait()
        assert database.committed == ()
    finally:
        database.release.set()
        await runtime.stop(1)
    assert database.committed == ("caller:inbound", "asynchronous")


@pytest.mark.asyncio
async def test_deferred_capture_freezes_pricing_before_configuration_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_logging_usage
    from litellm.types.utils import ModelResponse, Usage
    from litellm.utils import dispatch_success_logging

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    logger.defer_async_logging = True
    logger.custom_cost_per_second = 0.25
    monkeypatch.setattr(litellm, "cost_discount_config", {"openai": 0.1})
    response: Final = ModelResponse(id="deferred", usage=Usage(prompt_tokens=3))
    try:
        dispatch_success_logging(logger, response, NOW, NOW, False, False)
        assert database.rows == ()
        logger.custom_cost_per_second = 9
        monkeypatch.setattr(litellm, "cost_discount_config", {"openai": 0.9})
        assert runtime.meter is not None
        capture_logging_usage(runtime.meter, logger, response, NOW, NOW)
        await runtime.meter.flush()
    finally:
        await runtime.stop(1)
    pricing: Final = json.loads(database.rows[-1].envelope)["pricing"]
    assert pricing["logging"]["custom_cost_per_second"] == 0.25
    assert pricing["adjustments"]["discount"] == 0.1


@pytest.mark.asyncio
async def test_synchronous_internal_capture_reserves_capacity_before_scheduling() -> None:
    from litellm._internal_context import internal_sub_call
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(max_pending_records=2, batch_size=2))
    logger: Final = _logger()

    def capture_child() -> None:
        with internal_sub_call("caller"):
            logger.capture_usage_before_logging(ModelResponse(id="child", usage=Usage(prompt_tokens=2)), NOW, NOW)

    try:
        await asyncio.to_thread(capture_child)
        assert runtime.meter is not None
        assert runtime.meter.pending_records == 1
        receipt: Final = await asyncio.to_thread(runtime.meter.submit, (_write("second"),))
        await database.entered.wait()
        with pytest.raises(UsageCaptureError, match="capacity"):
            await asyncio.to_thread(runtime.meter.submit, (_write("overflow"),))
        database.release.set()
        await receipt
    finally:
        database.release.set()
        await runtime.stop(1)
    assert database.committed == ("child:included", "second")


@pytest.mark.asyncio
@pytest.mark.parametrize("has_metered_usage", [False, True])
async def test_realtime_without_provider_token_usage_does_not_invent_zero(has_metered_usage: bool) -> None:
    from litellm.constants import REALTIME_METERED_USAGE_KEY
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.types.llms.xai import RealtimeMeteredUsage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger(CallTypes.arealtime.value)
    if has_metered_usage:
        logger.model_call_details[REALTIME_METERED_USAGE_KEY] = RealtimeMeteredUsage(True, 0, 0, 0)
    capture_logging_usage(meter, logger, [], NOW, NOW)
    await meter.close()
    usage: Final = json.loads(database.rows[-1].envelope)["usage"]
    assert usage == (
        {
            "metered_usage": {
                "push_to_talk": True,
                "audio_seconds_sent": 0,
                "audio_seconds_received": 0,
                "text_messages": 0,
            }
        }
        if has_metered_usage
        else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("committed_before_failure", [False, True])
async def test_database_recovery_retries_frozen_usage_without_callback_or_duplicate_rows(
    committed_before_failure: bool,
) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = StoredUsage()
    failures: Final = asyncio.Queue[None]()
    failures.put_nowait(None)
    retrying: Final = asyncio.Event()
    recovered: Final = asyncio.Event()
    delays: Final = asyncio.Queue[float]()

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        if not failures.empty():
            failures.get_nowait()
            if committed_before_failure:
                await database.insert(writes)
            raise ConnectionError("database unavailable or acknowledgement lost")
        await database.insert(writes)

    async def retry_delay(delay: float) -> None:
        delays.put_nowait(delay)
        retrying.set()
        await recovered.wait()

    meter: Final = DurableUsageMeter(
        persist, UsageCaptureSettings(max_pending_records=2), wait_for_batch=_manual_flush, retry_delay=retry_delay
    )
    kwargs: Final = {
        "call_type": "acompletion",
        "litellm_call_id": "caller",
        "model": "capture-model",
        "litellm_params": {},
    }
    response: Final = {"id": "provider", "usage": {"total_tokens": 3}}
    first: Final = capture_callback(meter, kwargs, response, NOW, NOW)
    reserved: Final = meter.pending_bytes
    flushing: Final = asyncio.create_task(meter.flush())
    await retrying.wait()
    assert not first.receipt.done()
    assert (meter.pending_records, meter.pending_bytes) == (2, reserved)
    with pytest.raises(UsageCaptureError, match="capacity"):
        meter.submit((_write("overflow"),))
    recovered.set()
    await flushing
    await first.receipt
    duplicate: Final = capture_callback(meter, {**kwargs, "_accounting_capture": first}, response, NOW, NOW)
    await meter.close()
    assert duplicate is first
    assert database.rows == first.writes
    assert delays.get_nowait() == 0.2
    assert (meter.pending_records, meter.pending_bytes) == (0, 0)


@pytest.mark.asyncio
async def test_cut_stream_keeps_original_model_usage_as_part_of_the_parent_aggregate() -> None:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.router import Router
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger(stream=True)
    stream: Final = CustomStreamWrapper(completion_stream=iter(()), model="capture-model", logging_obj=logger)
    partial: Final = ModelResponse(
        id="cut", model="capture-model", usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    )
    try:
        Router._price_cut_partial(stream, partial)
    finally:
        await runtime.stop(1)
    assert len(database.rows) == 1
    payload: Final = json.loads(database.rows[0].envelope)
    assert payload["event"]["event_id"] == "cut:included"
    assert payload["parent_request_id"] == "caller"
    assert payload["charge_eligible"] is False
    assert payload["usage"]["total_tokens"] == 5
    assert payload["pricing"]["model_map_information"]["model_map_key"] == "capture-model"


@pytest.mark.asyncio
async def test_direct_mcp_logging_captures_tool_pricing_and_identity_without_arguments() -> None:
    from mcp.types import CallToolResult, TextContent

    from litellm.accounting import DimensionName
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger(CallTypes.call_mcp_tool.value)
    logger.model_call_details["mcp_tool_call_metadata"] = {
        "name": "lookup",
        "namespaced_tool_name": "server.lookup",
        "arguments": {"query": "private query"},
        "mcp_server_cost_info": {"default_cost_per_query": 0.2, "tool_name_to_cost_per_query": {"lookup": 0.3}},
    }
    try:
        await logger.async_success_handler(
            CallToolResult(content=[TextContent(type="text", text="private result")]), NOW, NOW
        )
    finally:
        await runtime.stop(1)
    outbound: Final = database.rows[-1]
    assert any(d.name is DimensionName.TOOL and d.value == "server.lookup" for d in outbound.event.dimensions)
    payload: Final = json.loads(outbound.envelope)
    assert payload["pricing"]["tool"] == {"name": "lookup", "cost_per_query": 0.3}
    assert payload["usage"] is None
    assert all(secret not in outbound.envelope for secret in ("private query", "private result"))


@pytest.mark.asyncio
async def test_callback_overload_is_visible_without_suppressing_legacy_tracking(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, captured_callback_kwargs

    database: Final = CommitBarrier()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(max_pending_records=2, batch_size=2))
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    kwargs: Final = {"call_type": "acompletion", "litellm_call_id": "one", "litellm_params": {}}
    try:
        admitted: Final = captured_callback_kwargs(kwargs, {"id": "one"}, NOW, NOW)
        await database.entered.wait()
        rejected: Final = captured_callback_kwargs({**kwargs, "litellm_call_id": "two"}, {"id": "two"}, NOW, NOW)
        assert "_accounting_capture" in admitted
        assert "_accounting_capture" not in rejected
        assert "Usage capture failed before legacy cost tracking" in caplog.text
    finally:
        database.release.set()
        await runtime.stop(1)
    assert database.committed == ("one:inbound", "one")


@pytest.mark.parametrize(
    ("call_type", "streaming", "background"),
    [
        ("acompletion", True, False),
        ("aresponses", False, True),
        ("arealtime", False, False),
    ],
)
def test_strict_ack_rejects_requests_without_a_terminal_response_boundary(
    call_type: str, streaming: bool, background: bool
) -> None:
    from fastapi import HTTPException

    from litellm.proxy.spend_tracking.usage_capture import validate_sync_ack_request

    validate_sync_ack_request({}, call_type, streaming, background)
    with pytest.raises(HTTPException) as error:
        validate_sync_ack_request({"accounting_sync_ack": True}, call_type, streaming, background)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_new_connection_turn_does_not_reuse_the_previous_turns_capture() -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    try:
        logger.capture_usage_before_logging(ModelResponse(id="first", usage=Usage(prompt_tokens=2)), NOW, NOW)
        next_turn: Final = logger.copy_for_new_call("second-caller", NOW)
        next_turn.capture_usage_before_logging(ModelResponse(id="second", usage=Usage(prompt_tokens=4)), NOW, NOW)
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == (
        "caller:inbound",
        "first",
        "second-caller:inbound",
        "second",
    )
    assert json.loads(database.rows[-1].envelope)["usage"]["prompt_tokens"] == 4


@pytest.mark.asyncio
async def test_billed_failure_is_captured_before_legacy_persistence_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    request: Final = {
        "call_type": "acompletion",
        "litellm_call_id": "failed",
        "model": "capture-model",
        "start_time": NOW,
        "combined_usage_object": Usage(prompt_tokens=3, total_tokens=3),
        "response_cost": 0.2,
        "litellm_params": {},
    }
    try:
        await ProxyDBLogger().async_post_call_failure_hook(request, RuntimeError("provider failed"), UserAPIKeyAuth())
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == ("failed:inbound", "failed")
    payload: Final = json.loads(database.rows[-1].envelope)
    assert payload["event"]["outcome"] == "failed"
    assert payload["usage"]["total_tokens"] == 3
    assert payload["charge_eligible"] is True
    assert payload["pricing"]["reported_cost"] == 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("pages", [None, 0, 3])
async def test_prepared_ocr_capture_preserves_reported_pages(pages: int | None) -> None:
    from litellm.llms.base_llm.ocr.transformation import OCRResponse, OCRUsageInfo
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger("aocr")
    response: Final = OCRResponse(pages=[], model="capture-model", usage_info=OCRUsageInfo(pages_processed=pages))
    logger.prepare_response_cost(response, NOW)
    capture_logging_usage(meter, logger, response, NOW, NOW)
    await meter.close()
    envelope: Final = json.loads(database.rows[-1].envelope)
    assert (None if envelope["usage"] is None else envelope["usage"]["pages_processed"]) == pages
    assert envelope["usage_source"] == ("unknown" if pages is None else "measured")
    assert response.usage_info is not None and response.usage_info.pages_processed == pages


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [None, {"total_tokens": 0}, {"total_tokens": 3}])
async def test_batch_capture_provenance_tracks_observations_not_envelope_presence(
    measured: Mapping[str, int] | None,
) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    usage: Final = {
        "total_tokens": 0 if measured is None else measured["total_tokens"],
        "batch_usage_by_model": {"missing-model": None, "other-model": measured},
        "batch_unknown_usage_requests": 2 if measured is None else 1,
    }
    capture_callback(
        meter,
        {"call_type": "aretrieve_batch", "litellm_call_id": "caller", "litellm_params": {}},
        {"id": "batch", "usage": usage},
        NOW,
        NOW,
    )
    await meter.close()
    envelope: Final = json.loads(database.rows[-1].envelope)
    assert envelope["usage_source"] == ("unknown" if measured is None else "measured")
    assert envelope["usage"]["batch_usage_by_model"] == usage["batch_usage_by_model"]
    assert database.rows[-1].event.metrics == (
        None if measured is None else (Metric(name=MetricName.TOTAL_TOKENS, value=Decimal(measured["total_tokens"])),)
    )


@pytest.mark.asyncio
async def test_ack_admission_failure_does_not_turn_provider_success_into_failed_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings(batch_size=2, max_pending_records=2))
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(proxy_server, "general_settings", {"accounting_sync_ack": True})
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    meter: Final = runtime.meter
    assert meter is not None
    logger: Final = _logger()
    try:
        meter.submit((_write("occupied-one"), _write("occupied-two")))
        with pytest.raises(HTTPException) as error:
            await capture_proxy_response(logger, ModelResponse(id="caller", usage=Usage(prompt_tokens=3)))
        assert error.value.status_code == 503
        await meter.flush()
        request: Final = {**logger.model_call_details, "litellm_logging_obj": logger, "start_time": NOW}
        await ProxyDBLogger().async_post_call_failure_hook(request, error.value, UserAPIKeyAuth())
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == ("occupied-one", "occupied-two")


@pytest.mark.asyncio
async def test_response_ack_timeout_preserves_success_and_later_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=2))
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {"accounting_sync_ack": True})
    logger: Final = _logger()

    async def expired_deadline(future: object, timeout: float) -> None:
        raise TimeoutError("controlled response deadline")

    try:
        with monkeypatch.context() as clock:
            clock.setattr(asyncio, "wait_for", expired_deadline)
            with pytest.raises(HTTPException) as error:
                await capture_proxy_response(logger, ModelResponse(id="provider", usage=Usage(prompt_tokens=3)))
        assert error.value.status_code == 503
        await database.entered.wait()
        captured: Final = logger.model_call_details["_accounting_capture"]
        request: Final = {**logger.model_call_details, "litellm_logging_obj": logger, "start_time": NOW}
        await ProxyDBLogger().async_post_call_failure_hook(request, error.value, UserAPIKeyAuth())
        database.release.set()
        await captured.receipt
        assert tuple(row.event.outcome for row in captured.writes) == (Outcome.SUCCEEDED, Outcome.SUCCEEDED)
        assert database.committed == ("caller:inbound", "provider")
        assert database.transactions == 1
    finally:
        await runtime.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["generate_content", "agenerate_content"])
@pytest.mark.parametrize("tokens", [None, 0, 7])
async def test_prepared_native_google_capture_preserves_reported_usage(
    call_type: str, tokens: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    import litellm
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger(call_type)
    monkeypatch.setitem(
        litellm.model_cost,
        "capture-model",
        {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2, "litellm_provider": "openai", "mode": "chat"},
    )
    reported: Final = (
        {"trafficType": "ON_DEMAND"}
        if tokens is None
        else {
            "promptTokenCount": tokens,
            "candidatesTokenCount": tokens,
            "totalTokenCount": 2 * tokens,
            "cachedContentTokenCount": tokens,
            "promptTokensDetails": [{"modality": "DOCUMENT", "tokenCount": tokens}],
            "cacheTokensDetails": [{"modality": "document", "token_count": tokens}],
        }
    )
    response: Final = {
        "responseId": "native-google",
        "candidates": [{"content": {"parts": [{"text": "answer"}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": reported,
    }
    native_response: Final = httpx.Response(200, json=response)
    logger.model_call_details["httpx_response"] = native_response
    prepared: Final = logger.prepare_response_cost(response, NOW)
    assert prepared is not None
    capture_logging_usage(meter, logger, response, NOW, NOW)
    await meter.close()
    envelope: Final = json.loads(database.rows[-1].envelope)
    assert (None if envelope["usage"] is None else envelope["usage"]["total_tokens"]) == (
        None if tokens is None else 2 * tokens
    )
    assert (None if envelope["usage"] is None else envelope["usage"]["prompt_tokens_details"]["text_tokens"]) == tokens
    assert (
        None
        if envelope["usage"] is None
        else envelope["usage"]["prompt_tokens_details"]["cached_tokens_details"]["text_tokens"]
    ) == tokens
    assert envelope["usage_source"] == ("unknown" if tokens is None else "measured")
    assert response == native_response.json()
