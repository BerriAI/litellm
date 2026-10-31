import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from litellm.accounting import Metric, MetricName, OutboundUsageEvent, Outcome, Unpriced
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
        unique: Final = {row.event.event_id: row for row in writes}
        if any(unique[row.event.event_id] != row for row in writes):
            raise AccountingConflictError("Submission contains conflicting content")
        self.rows += tuple(row for identity, row in unique.items() if identity not in existing)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["parent\x00value", "pärent-✓-日本", r"\u0000"])
async def test_capture_normalizes_database_text_without_poisoning_neighbors(text: str) -> None:
    from litellm.proxy.spend_tracking.usage_capture import PricingCapture, capture_callback

    database: Final = StoredUsage()
    batches: Final = asyncio.Queue[tuple[UsageWrite, ...]]()

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        assert all("\x00" not in (json.loads(write.envelope).get("parent_request_id") or "") for write in writes)
        batches.put_nowait(writes)
        await database.insert(writes)

    meter: Final = DurableUsageMeter(persist, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    nested: Final = json.loads("[" * 100 + "3" + "]" * 100)
    kwargs: Final = {
        "call_type": "acompletion",
        "litellm_call_id": "caller",
        "model": "capture-model",
        "search_context_size": text,
        "_accounting_pricing_snapshot": PricingCapture(
            "{}", "{}", json.dumps({"request": {"search_context_size": text}, "nested": nested})
        ),
        "litellm_params": {"metadata": {"parent_request_id": text}},
    }
    response: Final = {"id": "provider", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    captured: Final = capture_callback(meter, kwargs, response, NOW, NOW)
    neighbor: Final = capture_callback(
        meter,
        {**kwargs, "litellm_call_id": "neighbor", "litellm_params": {}},
        {**response, "id": "neighbor-provider"},
        NOW,
        NOW,
    )
    try:
        await meter.flush()
        await asyncio.gather(captured.receipt, neighbor.receipt)
        assert batches.qsize() == 1
        assert len(batches.get_nowait()) == 4
        envelope: Final = json.loads(captured.writes[-1].envelope)
        assert envelope["parent_request_id"] == text.replace("\x00", "")
        assert envelope["pricing"]["request"]["search_context_size"] == text.replace("\x00", "")
        assert envelope["pricing"]["nested"] == nested
        assert envelope["event"] == captured.writes[-1].event.model_dump(mode="json")
        assert envelope["usage"] == response["usage"]
        assert kwargs["litellm_params"] == {"metadata": {"parent_request_id": text}}
        repeated: Final = capture_callback(meter, kwargs, response, NOW, NOW)
        await meter.flush()
        await repeated.receipt
        assert repeated.writes == captured.writes
        assert len(database.rows) == 4
    finally:
        await meter.stop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["identity", "context"])
async def test_invalid_capture_text_is_rejected_before_admission(invalid_field: str) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    kwargs: Final = {"call_type": "acompletion", "litellm_call_id": "caller", "litellm_params": {}}
    try:
        accepted: Final = capture_callback(meter, kwargs, {"id": "provider"}, NOW, NOW)
        invalid: Final = (
            {**kwargs, "litellm_call_id": "caller\x00"}
            if invalid_field == "identity"
            else {**kwargs, "litellm_params": {"metadata": {"parent_request_id": "parent\ud800"}}}
        )
        with pytest.raises((UsageCaptureError, ValueError)):
            capture_callback(meter, invalid, {"id": "other"}, NOW, NOW)
        assert meter.pending_records == 2
        await meter.flush()
        await accepted.receipt
        assert tuple(write.event.event_id for write in database.rows) == ("caller:inbound", "provider")
    finally:
        await meter.stop(0)


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
@pytest.mark.parametrize("terminal", [False, True])
async def test_competing_writers_reuse_only_terminal_pricing_snapshots(terminal: bool) -> None:
    from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
    from litellm.proxy.spend_tracking.usage_capture import UsageBatchWriter, capture_callback
    from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

    database: Final = StoredUsage()
    attempts: Final = asyncio.Queue[tuple[UsageWrite, ...]]()
    competing: Final = asyncio.Event()

    async def insert(writes: tuple[UsageWrite, ...]) -> None:
        attempts.put_nowait(writes)
        if attempts.qsize() >= 2:
            competing.set()
        await competing.wait()
        await database.insert(writes)

    meters: Final = tuple(
        DurableUsageMeter(UsageBatchWriter(insert, database.load).write, UsageCaptureSettings()) for _ in range(2)
    )
    kwargs: Final = {
        "call_type": "aget_responses" if terminal else "acompletion",
        "litellm_call_id": "caller",
        "model": "capture-model",
        "litellm_params": {
            "metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN}
            if terminal
            else {}
        },
    }
    response: Final = {
        "id": "job",
        "status": "completed",
        "background": terminal,
        "created_at": NOW.timestamp(),
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    }
    captured: Final = tuple(
        capture_callback(meter, {**kwargs, "response_cost": price}, response, NOW, NOW)
        for meter, price in zip(meters, (2.0, 9.0), strict=True)
    )
    try:
        await asyncio.gather(*(meter.flush() for meter in meters))
        results: Final = await asyncio.gather(*(capture.receipt for capture in captured), return_exceptions=True)
        assert sum(result is None for result in results) == (2 if terminal else 1)
        assert sum(isinstance(result, UsageCaptureError) for result in results) == (0 if terminal else 1)
        assert database.rows in tuple(capture.writes for capture in captured)
        assert len(database.rows) == (1 if terminal else 2)
        assert attempts.qsize() == (3 if terminal else 2)
    finally:
        await asyncio.gather(*(meter.stop(0) for meter in meters))


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
@pytest.mark.parametrize("call_type", ["acompletion", "atext_completion", "aembedding", "aresponses"])
async def test_supported_sync_ack_waits_for_commit(call_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import ConfigGeneralSettings
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=2))
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(proxy_server, "general_settings", {"accounting_sync_ack": True})
    deadlines: Final = asyncio.Queue[float | None]()
    original_timeout: Final = asyncio.timeout

    def timeout_with_deadline(delay: float | None) -> asyncio.Timeout:
        deadlines.put_nowait(delay)
        return original_timeout(delay)

    monkeypatch.setattr(asyncio, "timeout", timeout_with_deadline)
    response: Final = asyncio.create_task(
        capture_proxy_response(
            _logger(call_type),
            ModelResponse(id="provider", usage=Usage(prompt_tokens=3)),
        )
    )
    try:
        await database.entered.wait()
        deadline: Final = deadlines.get_nowait()
        assert deadline == ConfigGeneralSettings().accounting_sync_ack_timeout_seconds
        assert deadline is not None and deadline > UsageCaptureSettings().flush_interval_seconds + 10
        assert not response.done()
        assert database.committed == ()
        database.release.set()
        await response
        assert database.committed == ("caller:inbound", "provider")
    finally:
        database.release.set()
        await runtime.stop(1)
        await response


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
        for _ in range(2):
            logger.record_partial_usage_for_failure(Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7), 0.4)
            logger.carry_failed_attempt_usage()
        logger.capture_usage_before_logging(
            ModelResponse(id="fallback", usage=Usage(prompt_tokens=3, total_tokens=3)), NOW, NOW
        )
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == (
        "caller-attempt-0",
        "caller-attempt-1",
        "caller:inbound",
        "fallback",
    )
    failed: Final = json.loads(database.rows[0].envelope)
    assert failed["usage"]["total_tokens"] == 7
    assert failed["parent_request_id"] == "caller"
    assert failed["request_count_eligible"] is False
    assert failed["charge_eligible"] is True
    assert failed["event"]["retry_of_attempt_id"] is None
    assert failed["event"]["outcome"] == "failed"
    assert failed["pricing"]["reported_cost"] == 0.4
    assert all(json.loads(row.envelope)["included_attempt_ids"] == [] for row in database.rows[:2])
    successful: Final = json.loads(database.rows[-1].envelope)
    assert successful["related_attempt_ids"] == ["caller-attempt-0", "caller-attempt-1"]
    assert successful["included_attempt_ids"] == []
    assert successful["usage"]["total_tokens"] == 3
    assert successful["charge_eligible"] is True
    assert database.rows[2].event.outcome is Outcome.SUCCEEDED


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
        for start, end in ((None, NOW), (NOW, None)):
            missing_timestamp: Final = await captured_callback_kwargs(kwargs, {"id": "one"}, start, end)
            assert missing_timestamp == kwargs
        assert "Usage capture requires callback start and end timestamps" in caplog.text
        admitted: Final = await captured_callback_kwargs(kwargs, {"id": "one"}, NOW, NOW)
        await database.entered.wait()
        rejected: Final = await captured_callback_kwargs({**kwargs, "litellm_call_id": "two"}, {"id": "two"}, NOW, NOW)
        assert "_accounting_capture" in admitted
        assert "_accounting_capture" not in rejected
        assert "Usage capture failed before legacy cost tracking" in caplog.text
    finally:
        database.release.set()
        await runtime.stop(1)
    assert database.committed == ("one:inbound", "one")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "streaming", "background"),
    [("acompletion", True, False), ("aresponses", False, True), ("aocr", False, False)],
)
@pytest.mark.parametrize("overloaded", [False, True])
async def test_async_capture_preserves_response_without_commit_ack(
    call_type: str, streaming: bool, background: bool, overloaded: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=2, max_pending_records=2))
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {
            "accounting_sync_ack": True,
            "accounting_sync_ack_timeout_seconds": 0,
        },
    )
    meter: Final = runtime.meter
    assert meter is not None
    if overloaded:
        meter.submit((_write("occupied-one"), _write("occupied-two")))
    try:
        await capture_proxy_response(
            _logger(call_type, stream=streaming),
            ModelResponse(id="provider", usage=Usage(prompt_tokens=3)),
            background=background,
        )
        assert database.committed == ()
        assert meter.pending_records == 2
    finally:
        database.release.set()
        await runtime.stop(1)
    assert database.committed == (("occupied-one", "occupied-two") if overloaded else ("caller:inbound", "provider"))


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
    monkeypatch.setattr(
        proxy_server, "general_settings", {"accounting_sync_ack": True, "accounting_sync_ack_timeout_seconds": 0.01}
    )
    logger: Final = _logger()

    try:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [False, True])
async def test_retry_links_only_admitted_usage_and_preserves_later_measurement(rejected: bool) -> None:
    from litellm.litellm_core_utils.litellm_logging import UsageObservation
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings(max_pending_records=4))
    logger: Final = _logger()
    assert runtime.meter is not None
    try:
        if rejected:
            runtime.meter.submit(tuple(_write(f"occupied-{index}") for index in range(4)))
        logger.model_call_details["_accounting_usage_observed"] = UsageObservation.UNKNOWN
        logger.record_partial_usage_for_failure(Usage(prompt_tokens=5, total_tokens=5), 0.4)
        logger.carry_failed_attempt_usage()
        await runtime.meter.flush()
        logger.capture_usage_before_logging(ModelResponse(id="success", usage=Usage(prompt_tokens=3)), NOW, NOW)
    finally:
        await runtime.stop(1)
    successful: Final = json.loads(next(row.envelope for row in database.rows if row.event.event_id == "success"))
    assert successful.get("related_attempt_ids", successful["included_attempt_ids"]) == (
        [] if rejected else ["caller-attempt-0"]
    )
    assert successful["usage"]["prompt_tokens"] == 3
    if not rejected:
        failed: Final = json.loads(
            next(row.envelope for row in database.rows if row.event.event_id == "caller-attempt-0")
        )
        assert failed["usage"]["prompt_tokens"] == 5
        assert failed["usage_source"] == "measured"


@pytest.mark.asyncio
@pytest.mark.parametrize("current_tokens", [None, 0, 3])
async def test_final_failure_captures_current_attempt_without_legacy_retry_aggregate(
    current_tokens: int | None,
) -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_callback
    from litellm.proxy.utils import _failure_fields_to_lift
    from litellm.types.utils import Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    try:
        logger.record_partial_usage_for_failure(Usage(prompt_tokens=5, total_tokens=5), 0.4)
        logger.carry_failed_attempt_usage()
        if current_tokens is not None:
            logger.record_partial_usage_for_failure(
                Usage(prompt_tokens=current_tokens, total_tokens=current_tokens), 0.2
            )
        lifted: Final = _failure_fields_to_lift({"litellm_logging_obj": logger}, provider_billed=True)
        assert lifted["combined_usage_object"].total_tokens == 5 + (current_tokens or 0)
        assert runtime.meter is not None
        capture_callback(
            runtime.meter,
            {**logger.model_call_details, **lifted, "log_event_type": "failed_api_call"},
            None,
            NOW,
            NOW,
        )
    finally:
        await runtime.stop(1)
    final: Final = json.loads(next(row.envelope for row in database.rows if row.event.event_id == "caller"))
    assert (None if final["usage"] is None else final["usage"]["total_tokens"]) == current_tokens
    assert final["usage_source"] == ("unknown" if current_tokens is None else "measured")
    assert final["pricing"]["reported_cost"] == (None if current_tokens is None else 0.2)
    assert final["included_attempt_ids"] == []
    assert final["related_attempt_ids"] == ["caller-attempt-0"]


@pytest.mark.asyncio
async def test_estimated_final_attempt_keeps_its_provenance_after_measured_retry() -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_callback
    from litellm.proxy.utils import _failure_fields_to_lift
    from litellm.types.utils import Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    logger.messages = [{"role": "user", "content": "hello"}]
    logger.model_call_details["messages"] = logger.messages
    try:
        logger.record_partial_usage_for_failure(Usage(prompt_tokens=5, total_tokens=5), 0.4)
        logger.carry_failed_attempt_usage()
        logger.model_call_details["current_attempt_api_call_start_time"] = NOW
        lifted: Final = _failure_fields_to_lift({"litellm_logging_obj": logger}, provider_billed=True)
        assert runtime.meter is not None
        capture_callback(runtime.meter, {**logger.model_call_details, **lifted}, None, NOW, NOW)
    finally:
        await runtime.stop(1)
    final: Final = json.loads(next(row.envelope for row in database.rows if row.event.event_id == "caller"))
    assert final["usage_source"] == "estimated"
    assert final["usage"]["total_tokens"] > 0
    assert lifted["combined_usage_object"].total_tokens == final["usage"]["total_tokens"] + 5
    assert final["pricing"]["reported_cost"] == 0


@pytest.mark.asyncio
async def test_parent_receipt_cannot_acknowledge_when_a_linked_attempt_conflicts() -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    try:
        logger.record_partial_usage_for_failure(Usage(prompt_tokens=5), 0.4)
        logger.carry_failed_attempt_usage()
        assert runtime.meter is not None
        await runtime.meter.flush()
        other: Final = _logger()
        other.record_partial_usage_for_failure(Usage(prompt_tokens=8), 0.8)
        other.carry_failed_attempt_usage()
        other.capture_usage_before_logging(ModelResponse(id="success", usage=Usage(prompt_tokens=3)), NOW, NOW)
        captured: Final = other.model_call_details["_accounting_capture"]
        await runtime.meter.flush()
        with pytest.raises(UsageCaptureError, match="dependency"):
            await captured.receipt
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == ("caller-attempt-0",)


@pytest.mark.asyncio
async def test_unserializable_observation_does_not_interrupt_retry_or_legacy_usage() -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = StoredUsage()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.insert, UsageCaptureSettings())
    logger: Final = _logger()
    usage: Final = Usage(prompt_tokens=5, total_tokens=5, opaque_usage=object())
    try:
        logger.record_partial_usage_for_failure(usage, 0.4)
        assert logger.model_call_details["combined_usage_object"] is usage
        logger.carry_failed_attempt_usage()
        assert logger.model_call_details["failed_attempts_usage"] == ((usage, 0.4),)
        logger.capture_usage_before_logging(ModelResponse(id="success", usage=Usage(prompt_tokens=3)), NOW, NOW)
    finally:
        await runtime.stop(1)
    assert tuple(row.event.event_id for row in database.rows) == ("caller:inbound", "success")
    assert json.loads(database.rows[-1].envelope)["related_attempt_ids"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_committed", [False, True])
@pytest.mark.parametrize("batch_size", [2, 3])
async def test_retry_dependencies_do_not_resubmit_usage_or_expand_final_submission(
    prior_committed: bool, batch_size: int
) -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    database.release.set()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=batch_size, max_pending_records=3))
    logger: Final = _logger()
    try:
        logger.record_partial_usage_for_failure(Usage(prompt_tokens=5), 0.4)
        logger.carry_failed_attempt_usage()
        assert runtime.meter is not None
        if prior_committed:
            await runtime.meter.flush()
        logger.capture_usage_before_logging(ModelResponse(id="success", usage=Usage(prompt_tokens=3)), NOW, NOW)
        captured: Final = logger.model_call_details.get("_accounting_capture")
        assert captured is not None, "Final usage must be admitted independently of prior usage batch size"
        await runtime.meter.close()
        await captured.receipt
        assert runtime.meter.pending_records == 0
    finally:
        await runtime.stop(1)
    assert database.committed == ("caller-attempt-0", "caller:inbound", "success")
    final: Final = json.loads(captured.writes[-1].envelope)
    assert final["related_attempt_ids"] == ["caller-attempt-0"]
    assert final["usage"]["prompt_tokens"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    ("call_type", "response", "field", "quantity"),
    [
        ("arerank", {"id": "rerank", "results": [], "meta": {"tokens": {"input_tokens": 7}}}, "prompt_tokens", 7),
        ("aocr", {"id": "ocr", "usage_info": {"pages_processed": 3}}, "pages_processed", 3),
        (
            "acompletion",
            {"id": "google", "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7}},
            "prompt_tokens",
            7,
        ),
    ],
)
async def test_capture_preserves_native_usage_before_and_after_preparation(
    prepared: bool, call_type: str, response: dict[str, object], field: str, quantity: int
) -> None:
    from litellm._internal_context import internal_sub_call
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger(call_type)
    logger.model_call_details["response_cost"] = 0.25
    if prepared:
        assert logger.prepare_response_cost(response, NOW) is not None
    with internal_sub_call("parent"):
        capture_logging_usage(meter, logger, response, NOW, NOW)
    await meter.close()
    assert len(database.rows) == 1
    envelope: Final = json.loads(database.rows[0].envelope)
    assert envelope["usage"][field] == quantity
    assert envelope["usage_source"] == "measured"
    assert envelope["ownership"] == "included_in_parent"


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [(120000, 120000), (210000, 30000)])
@pytest.mark.parametrize("priced", [False, True])
async def test_batch_capture_retains_request_basis_for_later_tiered_pricing(
    tokens: tuple[int, int], priced: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm
    from litellm.batches.batch_utils import batch_cost_and_usage
    from litellm.cost_calculator import batch_cost_calculator
    from litellm.proxy.spend_tracking.usage_capture import (
        UsageBatchWriter,
        capture_batch_callback,
        capture_logging_usage,
    )
    from litellm.types.utils import Usage

    rates: Final = {
        "input_cost_per_token_batches": 0.001,
        "input_cost_per_token_above_200k_tokens_batches": 0.002,
        "output_cost_per_token_batches": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, "capture-model", rates if priced else {"litellm_provider": "openai"})
    rows: Final = [
        {
            "response": {
                "status_code": 200,
                "body": {
                    "model": "capture-model",
                    "usage": {"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n},
                },
            }
        }
        for n in tokens
    ]
    result: Final = batch_cost_and_usage(rows, "openai", model_info=rates if priced else {})
    database: Final = StoredUsage()
    writer: Final = UsageBatchWriter(database.insert, database.load)
    meter: Final = DurableUsageMeter(writer.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger("aretrieve_batch")
    logger.model_call_details["response_cost"] = result.cost
    capture_logging_usage(meter, logger, {"id": "batch", "status": "completed", "usage": result.usage}, NOW, NOW)
    await capture_batch_callback(
        meter,
        database.load,
        logger.model_call_details,
        {"id": "batch", "status": "completed", "usage": result.usage},
        NOW,
        NOW,
    )
    await meter.close()
    retained: Final = tuple(
        {"model": "capture-model", "usage": json.loads(write.envelope)["usage"]} for write in database.rows[1:]
    )
    assert sorted(row["usage"]["prompt_tokens"] for row in retained) == sorted(tokens)
    reconstructed: Final = sum(
        sum(batch_cost_calculator(Usage(**row["usage"]), row["model"], "openai", model_info=rates)) for row in retained
    )
    expected: Final = sum(n * (0.002 if n > 200000 else 0.001) for n in tokens)
    assert reconstructed == pytest.approx(expected)
    assert result.cost == pytest.approx(expected if priced else 0)
    assert database.rows[0].event.metrics is None
    assert json.loads(database.rows[0].envelope)["charge_eligible"] is False
    assert all(json.loads(write.envelope)["finalized_cost"] is None for write in database.rows[1:])


@pytest.mark.asyncio
async def test_large_batch_recovers_partial_commit_with_frozen_pricing_and_rejects_changed_usage() -> None:
    from litellm.proxy.spend_tracking.usage_capture import (
        PricingCapture,
        UsageBatchWriter,
        capture_batch_callback,
        capture_callback,
    )

    database: Final = StoredUsage()
    transactions: Final = asyncio.Queue[tuple[UsageWrite, ...]]()
    failure: Final = asyncio.Queue[bool]()
    failure.put_nowait(True)

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        assert sum(len(write.envelope.encode()) for write in writes) <= 65536
        assert len(writes) <= 32
        if len(database.rows) > 1 and not failure.empty():
            failure.get_nowait()
            raise RuntimeError("Interrupted batch persistence")
        await database.insert(writes)
        transactions.put_nowait(writes)

    writer: Final = UsageBatchWriter(persist, database.load)
    meter: Final = DurableUsageMeter(
        writer.write,
        UsageCaptureSettings(batch_size=32, max_pending_records=32, max_pending_bytes=65536),
        wait_for_batch=_manual_flush,
    )
    rows: Final = tuple(
        {
            "model": "capture-model" if index % 7 else None,
            "usage": {"prompt_tokens": index % 5 + 1} if index % 7 else None,
        }
        for index in range(1000)
    )
    response: Final = {"id": "large-batch", "status": "completed", "usage": {"batch_usage_by_request": rows}}
    pricing: Final = PricingCapture("{}", "{}", json.dumps({"batch_models": {"capture-model": {"rate": 1}}}))
    kwargs: Final = {
        "call_type": "aretrieve_batch",
        "litellm_call_id": "poll",
        "litellm_params": {},
        "_accounting_pricing_snapshot": pricing,
    }
    failed_database: Final = CommitBarrier(failure=RuntimeError("Earlier capture failed"))
    failed_database.release.set()
    prior_meter: Final = DurableUsageMeter(failed_database.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    prior: Final = capture_callback(
        prior_meter,
        {"call_type": "acompletion", "litellm_call_id": "earlier", "litellm_params": {}},
        {"id": "earlier"},
        NOW,
        NOW,
    )
    await prior_meter.close()
    with pytest.raises(RuntimeError, match="Earlier capture failed"):
        await capture_batch_callback(
            meter,
            database.load,
            {**kwargs, "_accounting_prior_captures": (prior,)},
            response,
            NOW,
            NOW,
        )
    assert database.rows == ()
    with pytest.raises(RuntimeError, match="Interrupted batch persistence"):
        await capture_batch_callback(meter, database.load, kwargs, response, NOW, NOW)
    partial: Final = database.rows
    assert 1 < len(partial) < len(rows) + 1
    changed_pricing: Final = PricingCapture("{}", "{}", json.dumps({"batch_models": {"capture-model": {"rate": 2}}}))
    retried: Final = await capture_batch_callback(
        meter,
        database.load,
        {**kwargs, "_accounting_pricing_snapshot": changed_pricing},
        {**response, "usage": {"batch_usage_by_request": tuple(reversed(rows))}},
        NOW,
        NOW,
    )
    assert retried.receipt.done() and retried.receipt.exception() is None
    assert len(database.rows) == len(rows) + 1
    assert database.rows[: len(partial)] == partial
    envelopes: Final = tuple(json.loads(write.envelope) for write in database.rows[1:])
    assert all(
        envelope["pricing"]["model_map_information"] == ({"rate": 1} if envelope["usage"] is not None else None)
        for envelope in envelopes
    )
    assert sum(envelope["usage"] is None for envelope in envelopes) == sum(row["usage"] is None for row in rows)
    assert all(envelope["request_count_eligible"] is False for envelope in envelopes)
    assert sum(envelope["usage"]["prompt_tokens"] for envelope in envelopes if envelope["usage"] is not None) == sum(
        row["usage"]["prompt_tokens"] for row in rows if row["usage"] is not None
    )
    before_conflict: Final = database.rows
    with pytest.raises(UsageCaptureError, match="identity conflicts"):
        await capture_batch_callback(
            meter,
            database.load,
            kwargs,
            {**response, "usage": {"batch_usage_by_request": rows[:-1]}},
            NOW,
            NOW,
        )
    await meter.close()
    assert database.rows == before_conflict
    assert transactions.qsize() > 2
    assert meter.pending_bytes == meter.pending_records == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["drain", "cancel", "failure"])
async def test_batch_callback_handoff_is_bounded_and_runtime_owns_persistence(
    outcome: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.usage_capture import CapturedUsage, UsageCaptureRuntime, captured_callback_kwargs

    database: Final = StoredUsage()
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("Persistence unavailable")
        await database.insert(writes)

    runtime: Final = UsageCaptureRuntime()
    runtime.start(persist, UsageCaptureSettings(batch_size=2, max_pending_records=2), load=database.load)
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    kwargs: Final = {"call_type": "aretrieve_batch", "litellm_call_id": "poll", "litellm_params": {}}
    response: Final = {
        "id": "batch",
        "status": "completed",
        "usage": {"batch_usage_by_request": ({"model": "model", "usage": {"prompt_tokens": 2}},) * 3},
    }
    callback: Final = asyncio.create_task(captured_callback_kwargs(kwargs, response, NOW, NOW))
    try:
        await entered.wait()
        assert callback.done(), "Batch persistence must not consume the legacy callback deadline"
        captured: Final = (await callback)["_accounting_capture"]
        assert isinstance(captured, CapturedUsage) and not captured.receipt.done()
        second: Final = await captured_callback_kwargs(kwargs, response, NOW, NOW)
        assert isinstance(second.get("_accounting_capture"), CapturedUsage)
        overflow: Final = await captured_callback_kwargs(kwargs, response, NOW, NOW)
        assert "_accounting_capture" not in overflow
        assert "Batch capture producer capacity exhausted" in caplog.text
        if outcome != "cancel":
            release.set()
        await runtime.stop(0 if outcome == "cancel" else 1)
        assert captured.receipt.done()
        assert captured.receipt.cancelled() is (outcome == "cancel")
        if outcome == "failure":
            assert isinstance(captured.receipt.exception(), RuntimeError)
            assert "Batch usage capture failed" in caplog.text
        assert len(database.rows) == (4 if outcome == "drain" else 0)
        with pytest.raises(UsageCaptureError, match="active ledger writer"):
            runtime.submit_batch(kwargs, response, NOW, NOW)
    finally:
        release.set()
        await asyncio.gather(callback, return_exceptions=True)
        await runtime.stop(1)


@pytest.mark.asyncio
async def test_batch_producers_share_byte_capacity_and_release_it_after_drain() -> None:
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime

    database: Final = StoredUsage()
    release: Final = asyncio.Event()
    runtime: Final = UsageCaptureRuntime()

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        await release.wait()
        assert runtime.meter is not None
        assert runtime.meter.pending_bytes <= 65536
        await database.insert(writes)

    runtime.start(persist, UsageCaptureSettings(batch_size=32, max_pending_bytes=65536), load=database.load)
    from litellm.batches.batch_utils import batch_cost_and_usage

    parsed: Final = batch_cost_and_usage(
        [
            {
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "capture-model",
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    },
                }
            }
            for _ in range(1000)
        ],
        "openai",
        model_info={},
    )
    rows: Final = ({"model": "capture-model", "usage": {"prompt_tokens": 2}},)
    response: Final = {"id": "batch", "status": "completed", "usage": parsed.usage}
    kwargs: Final = {"call_type": "aretrieve_batch", "litellm_call_id": "poll", "litellm_params": {}}
    try:
        large: Final = asyncio.Queue[asyncio.Future[None]]()
        for index in range(100):
            try:
                captured: Final = runtime.submit_batch(kwargs, {**response, "id": f"batch-{index}"}, NOW, NOW)
            except UsageCaptureError:
                break
            large.put_nowait(captured.receipt)
        large_count: Final = large.qsize()
        assert 0 < large_count < 100
        assert runtime.meter is not None and 0 < runtime.meter.pending_bytes <= 65536
        release.set()
        await asyncio.gather(*(large.get_nowait() for _ in range(large_count)))
        assert len(database.rows) == 1001 * large_count
        observations: Final = tuple(json.loads(write.envelope) for write in database.rows)
        assert sum(row["usage"].get("prompt_tokens", 0) for row in observations) == 2000 * large_count
        assert sum(row["usage"].get("completion_tokens", 0) for row in observations) == 1000 * large_count
        assert runtime.meter.pending_bytes == 0
        admitted: Final = asyncio.Queue[asyncio.Future[None]]()
        for index in range(100):
            try:
                small: Final = runtime.submit_batch(
                    kwargs,
                    {**response, "id": f"small-{index}", "usage": {"batch_usage_by_request": rows}},
                    NOW,
                    NOW,
                )
            except UsageCaptureError:
                break
            admitted.put_nowait(small.receipt)
        count: Final = admitted.qsize()
        assert 1 < count < 100
        await asyncio.gather(*(admitted.get_nowait() for _ in range(count)))
        assert len(database.rows) == 1001 * large_count + 2 * count
        assert runtime.meter.pending_bytes == 0
    finally:
        release.set()
        await runtime.stop(1)


@pytest.mark.asyncio
async def test_batch_replay_reserves_the_accepted_pricing_snapshot_size() -> None:
    from litellm.proxy.spend_tracking.usage_capture import PricingCapture, UsageBatchWriter, UsageCaptureRuntime

    database: Final = StoredUsage()
    writer: Final = UsageBatchWriter(database.insert, database.load)
    runtime: Final = UsageCaptureRuntime()
    runtime.start(writer.write, UsageCaptureSettings(max_pending_bytes=16384), load=database.load)
    response: Final = {
        "id": "batch",
        "status": "completed",
        "usage": {"batch_usage_by_request": ({"model": "model", "usage": {"prompt_tokens": 2}},)},
    }
    kwargs: Final = {"call_type": "aretrieve_batch", "litellm_call_id": "poll", "litellm_params": {}}
    pricing: Final = PricingCapture("{}", "{}", json.dumps({"batch_models": {"model": {"source": "x" * 1200}}}))
    try:
        initial: Final = runtime.submit_batch({**kwargs, "_accounting_pricing_snapshot": pricing}, response, NOW, NOW)
        await asyncio.shield(initial.receipt)
        accepted: Final = database.rows
        replay: Final = runtime.submit_batch(kwargs, response, NOW, NOW)
        await asyncio.shield(replay.receipt)
        assert database.rows == accepted and len(accepted) == 2
        assert runtime.meter is not None and runtime.meter.pending_bytes == 0
    finally:
        await runtime.stop(1)


class DelayedCostCompiler:
    def __init__(self) -> None:
        self.entered: Final = asyncio.Event()
        self.cancelled: Final = asyncio.Event()
        self.release: Final = asyncio.Event()

    async def compile(self, event: OutboundUsageEvent) -> Unpriced:
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return Unpriced(usage_event_id=event.event_id, reason="usage_unknown")


@pytest.mark.asyncio
async def test_sync_ack_deadline_bounds_compilation_after_usage_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.usage_capture import UsageCaptureRuntime, capture_proxy_response
    from litellm.types.utils import ModelResponse, Usage

    database: Final = CommitBarrier()
    database.release.set()
    compiler: Final = DelayedCostCompiler()
    runtime: Final = UsageCaptureRuntime()
    runtime.start(database.write, UsageCaptureSettings(batch_size=2), compiler=compiler)
    monkeypatch.setattr(proxy_server, "usage_capture", runtime)
    monkeypatch.setattr(
        proxy_server, "general_settings", {"accounting_sync_ack": True, "accounting_sync_ack_timeout_seconds": 0.1}
    )
    try:
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(
                capture_proxy_response(_logger(), ModelResponse(id="provider", usage=Usage(prompt_tokens=3))), 2
            )
        assert error.value.status_code == 503
        assert compiler.entered.is_set()
        assert compiler.cancelled.is_set()
        assert database.committed == ("caller:inbound", "provider")
    finally:
        compiler.release.set()
        await runtime.stop(1)
