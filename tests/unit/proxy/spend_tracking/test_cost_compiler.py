import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from litellm.accounting import Dimension, DimensionName, OutboundUsageEvent, Outcome, PricingSnapshot, Unpriced
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.spend_tracking.cost_compiler import price_usage
from litellm.types.utils import CallTypes

_AT: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _envelope() -> dict[str, object]:
    event: Final = OutboundUsageEvent(
        event_id="attempt",
        attempt_id="attempt",
        request_id="request",
        retry_of_attempt_id=None,
        call_type=CallTypes.acompletion,
        started_at=_AT,
        ended_at=_AT,
        outcome=Outcome.SUCCEEDED,
        dimensions=(
            Dimension(name=DimensionName.MODEL, value="openai/retained-model"),
            Dimension(name=DimensionName.PROVIDER, value="openai"),
        ),
        metrics=None,
    )
    return {
        "event": event.model_dump(mode="json"),
        "charge_eligible": True,
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "pricing": {
            "model_map_information": {
                "model_map_key": "openai/retained-model",
                "model_map_value": {
                    "input_cost_per_token": 0.01,
                    "output_cost_per_token": 0.02,
                    "litellm_provider": "openai",
                    "mode": "chat",
                },
            },
            "adjustments": {"discount": 0.1, "margin": {"fixed_amount": 0.008}},
        },
    }


def test_compiler_prices_retained_rates_without_registering_a_model() -> None:
    import litellm

    result: Final = price_usage(json.dumps(_envelope()))
    assert isinstance(result, PricingSnapshot)
    assert result.total_cost == Decimal(str((10 * 0.01 + 4 * 0.02) * 0.9 + 0.008))
    assert "openai/retained-model" not in litellm.model_cost


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("usage", None, "usage_unknown"),
        ("charge_eligible", False, "not_charge_owner"),
        ("pricing", {}, "pricing_inputs_missing"),
        ("finalized_cost", {"response_cost": 999}, "finalized_cost_mismatch"),
    ],
)
def test_compiler_does_not_invent_a_charge(field: str, value: object, reason: str) -> None:
    result: Final = price_usage(json.dumps({**_envelope(), field: value}))
    assert result == Unpriced(usage_event_id="attempt", reason=reason)


def _retained_envelope(
    *,
    rates: dict[str, object] | None = None,
    usage: dict[str, object] | None = None,
    logging: dict[str, object] | None = None,
    response_inputs: dict[str, object] | None = None,
    deployment: dict[str, object] | None = None,
    request: dict[str, object] | None = None,
) -> dict[str, object]:
    from pydantic import TypeAdapter

    original: Final = _envelope()
    pricing: Final = TypeAdapter(dict[str, object]).validate_python(original["pricing"])
    mapped: Final = TypeAdapter(dict[str, object]).validate_python(pricing["model_map_information"])
    base_rates: Final = TypeAdapter(dict[str, object]).validate_python(mapped["model_map_value"])
    return {
        **original,
        "usage": usage if usage is not None else original["usage"],
        "pricing": {
            **pricing,
            "model_map_information": {**mapped, "model_map_value": {**base_rates, **(rates or {})}},
            "logging": logging or {},
            "response_inputs": response_inputs or {},
            "deployment": deployment or {},
            "request": request or {},
            "adjustments": {"discount": None, "margin": None},
        },
    }


def test_retained_pricing_ignores_later_global_price_and_adjustment_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    captured: Final = json.dumps(_envelope())
    original: Final = price_usage(captured)
    monkeypatch.setattr(litellm, "model_cost", {"openai/retained-model": {"input_cost_per_token": 99.0}})
    monkeypatch.setattr(litellm, "cost_discount_config", {"openai": 0.99})
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": {"fixed_amount": 999.0}})
    recomputed: Final = price_usage(captured)
    assert isinstance(original, PricingSnapshot)
    assert recomputed == original
    assert litellm.model_cost["openai/retained-model"]["input_cost_per_token"] == 99.0


def test_repricing_recomputes_retained_usage_instead_of_returning_a_reported_total() -> None:
    from pydantic import TypeAdapter

    captured: Final = _retained_envelope()
    pricing: Final = TypeAdapter(dict[str, object]).validate_python(captured["pricing"])
    original: Final = price_usage(json.dumps({**captured, "pricing": {**pricing, "reported_cost": 999.0}}))
    changed: Final = price_usage(json.dumps({**captured, "usage": {"prompt_tokens": 20, "completion_tokens": 8}}))
    assert isinstance(original, PricingSnapshot)
    assert isinstance(changed, PricingSnapshot)
    assert float(original.total_cost) == pytest.approx(0.18)
    assert float(changed.total_cost) == pytest.approx(float(original.total_cost) * 2)


def test_retained_token_pricing_preserves_cache_reasoning_and_service_tier() -> None:
    captured: Final = _retained_envelope(
        rates={
            "input_cost_per_token_priority": 0.03,
            "output_cost_per_token_priority": 0.05,
            "cache_read_input_token_cost_priority": 0.002,
            "output_cost_per_reasoning_token_priority": 0.07,
        },
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
        response_inputs={"service_tier": "priority"},
    )
    result: Final = price_usage(json.dumps(captured))
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx(60 * 0.03 + 40 * 0.002 + 15 * 0.05 + 5 * 0.07)


@pytest.mark.parametrize("stamped_cost", [None, 0.9])
def test_upstream_reported_price_takes_precedence_without_custom_pricing(stamped_cost: float | None) -> None:
    result: Final = price_usage(
        json.dumps(
            _retained_envelope(
                logging={"provider_reported_cost": 0.037},
                usage={"prompt_tokens": 10, "completion_tokens": 4, "cost": stamped_cost},
            )
        )
    )
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx(0.037)


def test_per_request_pricing_overrides_upstream_reported_price() -> None:
    result: Final = price_usage(
        json.dumps(
            _retained_envelope(
                logging={
                    "provider_reported_cost": 0.037,
                    "custom_pricing": True,
                    "custom_cost_per_token": {"input_cost_per_token": 0.04, "output_cost_per_token": 0.06},
                }
            )
        )
    )
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx(10 * 0.04 + 4 * 0.06)


def test_retained_regional_pricing_preserves_data_residency_uplift() -> None:
    result: Final = price_usage(
        json.dumps(
            _retained_envelope(
                rates={"regional_processing_uplift_multiplier_eu": 1.1},
                deployment={"data_residency": "eu"},
            )
        )
    )
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx((10 * 0.01 + 4 * 0.02) * 1.1)


def test_retained_request_additional_cost_is_applied_once() -> None:
    result: Final = price_usage(json.dumps(_retained_envelope(request={"additional_response_cost": 0.3})))
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx(10 * 0.01 + 4 * 0.02 + 0.3)


def _capture_logger(call_type: str = "acompletion", *, stream: bool = False) -> Logging:
    from tests.unit.proxy.spend_tracking.test_usage_capture import _logger

    logger: Final = _logger(call_type, stream=stream)
    logger.set_request_pricing("capture-model", "openai", {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02})
    return logger


@pytest.mark.asyncio
async def test_real_capture_preserves_synchronous_finalization_and_guardrail_parity() -> None:
    from litellm.proxy.spend_tracking.cost_finalization import finalize_response_cost
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger()
    response: Final = ModelResponse(
        id="finalized", model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    )
    finalized: Final = await finalize_response_cost(logger, response, None, _AT, guardrail_cost=0.03)
    assert finalized is not None
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(finalized.amount)
        assert float(result.guardrail_cost or 0) == pytest.approx(0.03)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_real_stream_capture_only_prices_final_usage() -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, ModelResponseStream, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger(stream=True)
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, ModelResponseStream(id="chunk"), _AT, _AT)
        assert meter.pending_records == 0
        capture_logging_usage(
            meter,
            logger,
            ModelResponse(id="terminal", model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=4)),
            _AT,
            _AT,
        )
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(0.18)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_real_internal_capture_excludes_usage_owned_by_parent() -> None:
    from litellm._internal_context import internal_sub_call
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        with internal_sub_call("parent"):
            capture_logging_usage(
                meter,
                _capture_logger(),
                ModelResponse(id="child", model="capture-model", usage=Usage(prompt_tokens=10)),
                _AT,
                _AT,
            )
        await meter.close()
        assert price_usage(database.rows[-1].envelope) == Unpriced(
            usage_event_id=database.rows[-1].event.event_id, reason="not_charge_owner"
        )
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("already_calculated", (False, True))
async def test_real_mcp_capture_prices_retained_tool_fee_without_token_usage(already_calculated: bool) -> None:
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _logger(CallTypes.call_mcp_tool.value)
    logger.model_call_details["mcp_tool_call_metadata"] = {
        "name": "lookup",
        "namespaced_tool_name": "server.lookup",
        "mcp_server_cost_info": {"default_cost_per_query": 0.2, "tool_name_to_cost_per_query": {"lookup": 0.3}},
    }
    if already_calculated:
        logger.model_call_details["response_cost"] = 0.3
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, CallToolResult(content=[TextContent(type="text", text="ok")]), _AT, _AT)
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(0.3)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_real_agent_capture_prices_retained_per_query_fee() -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.agents import LiteLLMSendMessageResponse
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _logger(CallTypes.asend_message.value)
    logger.update_environment_variables(
        model="capture-model", custom_llm_provider="a2a", litellm_params={"cost_per_query": 0.4}, optional_params={}
    )
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, LiteLLMSendMessageResponse(id="agent"), _AT, _AT)
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(0.4)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_cost", (0.0, 0.123))
@pytest.mark.parametrize("stamped", (False, True))
@pytest.mark.parametrize("adjusted", (False, True))
async def test_real_capture_prices_upstream_cost_without_a_model_map(
    monkeypatch: pytest.MonkeyPatch, reported_cost: float, stamped: bool, adjusted: bool
) -> None:
    import litellm
    from litellm.litellm_core_utils.core_helpers import stamp_litellm_bill_on_usage
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    monkeypatch.setattr(litellm, "cost_discount_config", {"openai": 0.05} if adjusted else {})
    monkeypatch.setattr(
        litellm, "cost_margin_config", {"global": {"percentage": 0.1, "fixed_amount": 0.001}} if adjusted else {}
    )
    expected: Final = reported_cost * 0.95 * 1.1 + 0.001 if adjusted else reported_cost
    logger: Final = _logger()

    response: Final = ModelResponse(
        id="upstream", model="capture-model", usage=Usage(prompt_tokens=10, cost=reported_cost)
    )
    if stamped:
        stamp_litellm_bill_on_usage(response, response.usage, reported_cost + 0.9)
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(expected)
        assert result.upstream_reported_cost == Decimal(str(reported_cost))
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_real_failure_capture_preserves_incurred_charge(cancelled: bool) -> None:
    from fastapi import HTTPException

    from litellm.litellm_core_utils.litellm_logging import UsageObservation
    from litellm.proxy.spend_tracking.usage_capture import capture_callback, capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger()
    usage: Final = Usage(prompt_tokens=4, completion_tokens=0, total_tokens=4)
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(
            meter, logger, ModelResponse(model="capture-model", usage=usage), _AT, _AT, freeze_only=True
        )
        capture_callback(
            meter,
            {
                **logger.model_call_details,
                "litellm_logging_obj": logger,
                "combined_usage_object": usage,
                "_accounting_usage_observed": UsageObservation.KNOWN,
                "log_event_type": "failed_api_call",
            },
            HTTPException(499, "Client disconnected") if cancelled else RuntimeError("provider failed after billing"),
            _AT,
            _AT,
        )
        await meter.close()
        outbound: Final = database.rows[-1]
        assert outbound.event.outcome is (Outcome.CANCELLED if cancelled else Outcome.FAILED)
        result: Final = price_usage(outbound.envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(0.04)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_captured_price_replay_uses_original_off_peak_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    import litellm
    from litellm._internal_context import pinned_billing_time
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _logger()
    monkeypatch.setitem(
        litellm.model_cost,
        "capture-model",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "off_peak_pricing": {
                "hours_utc": "00:00-01:00",
                "input_cost_per_token": 0.005,
                "output_cost_per_token": 0.01,
            },
        },
    )
    response: Final = ModelResponse(model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=4))
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        with pinned_billing_time(_AT):
            source_cost: Final = logger.response_cost_calculator(response, ignore_stamped_cost=True)
            capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        with pinned_billing_time(_AT + timedelta(hours=3)):
            replay: Final = price_usage(database.rows[-1].envelope)
        assert source_cost == pytest.approx(0.09)
        assert isinstance(replay, PricingSnapshot)
        assert float(replay.total_cost) == pytest.approx(source_cost)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_real_realtime_capture_preserves_audio_and_text_token_prices() -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger(CallTypes.arealtime.value)
    logger.set_request_pricing(
        "capture-model",
        "openai",
        {
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "input_cost_per_audio_token": 0.03,
            "output_cost_per_audio_token": 0.04,
        },
    )
    events: Final = [
        {
            "type": "response.done",
            "response": {
                "id": "turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_token_details": {"text_tokens": 4, "audio_tokens": 6, "cached_tokens": 0},
                    "output_token_details": {"text_tokens": 2, "audio_tokens": 3},
                },
                "output": "private transcript excluded from accounting",
            },
        }
    ]
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, events, _AT, _AT)
        await meter.close()
        captured: Final = database.rows[-1].envelope
        assert "private transcript" not in captured
        result: Final = price_usage(captured)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(4 * 0.01 + 6 * 0.03 + 2 * 0.02 + 3 * 0.04)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_real_realtime_capture_preserves_metered_audio_and_message_fees() -> None:
    from litellm.constants import REALTIME_METERED_USAGE_KEY
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.llms.xai import RealtimeMeteredUsage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _logger(CallTypes.arealtime.value)
    logger.update_environment_variables(
        model="capture-model", custom_llm_provider="xai", litellm_params={}, optional_params={}
    )
    logger.set_request_pricing(
        "capture-model",
        "xai",
        {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "input_cost_per_second": 0.02,
            "input_cost_per_text_message": 0.1,
        },
    )
    logger.model_call_details[REALTIME_METERED_USAGE_KEY] = RealtimeMeteredUsage(True, 1.5, 2.5, 3)
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(
            meter,
            logger,
            [
                {
                    "type": "response.done",
                    "response": {
                        "id": "metered",
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    },
                }
            ],
            _AT,
            _AT,
        )
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx((1.5 + 2.5) * 0.02 + 3 * 0.1)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_real_batch_rows_keep_per_request_tiers_across_repeated_terminal_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.batches.batch_utils import batch_cost_and_usage
    from litellm.proxy.spend_tracking.usage_capture import (
        UsageBatchWriter,
        capture_batch_callback,
        capture_logging_usage,
    )
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    rates: Final = {
        "input_cost_per_token_batches": 0.001,
        "input_cost_per_token_above_200k_tokens_batches": 0.002,
        "output_cost_per_token_batches": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, "capture-model", rates)
    rows: Final = [
        {
            "response": {
                "status_code": 200,
                "body": {
                    "model": "capture-model",
                    "usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
                },
            }
        }
        for tokens in (10, 200001)
    ]
    billed: Final = batch_cost_and_usage(rows, "openai")
    response: Final = {"id": "batch", "status": "completed", "usage": billed.usage}
    database: Final = StoredUsage()
    writer: Final = UsageBatchWriter(database.insert, database.load)
    meter: Final = DurableUsageMeter(writer.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger("aretrieve_batch")
    logger.model_call_details["response_cost"] = billed.cost
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await capture_batch_callback(meter, database.load, logger.model_call_details, response, _AT, _AT)
        await meter.flush()
        first: Final = tuple(database.rows)
        await capture_batch_callback(meter, database.load, logger.model_call_details, response, _AT, _AT)
        await meter.close()
        assert tuple(database.rows) == first
        assert price_usage(database.rows[0].envelope) == Unpriced(
            usage_event_id=database.rows[0].event.event_id, reason="not_charge_owner"
        )
        costs: Final = tuple(price_usage(row.envelope) for row in database.rows[1:])
        assert all(isinstance(cost, PricingSnapshot) for cost in costs), costs
        assert sum(float(cost.total_cost) for cost in costs if isinstance(cost, PricingSnapshot)) == pytest.approx(
            10 * 0.001 + 200001 * 0.002
        )
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("transcription", (False, True))
async def test_realtime_capture_replays_sanitized_lifecycle_and_transcription_charges(transcription: bool) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger(CallTypes.arealtime.value)
    logger.set_request_pricing(
        "capture-model",
        "openai",
        {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "input_cost_per_second": 0.02,
        },
    )
    events: Final = (
        [
            {
                "type": "transcription_session.created",
                "session": {"input_audio_transcription": {"model": "capture-model"}},
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "audio",
                "transcript": "private transcript",
                "usage": {"type": "duration", "seconds": 2.5},
            },
        ]
        if transcription
        else [
            {"type": "session.started", "session": {"model": "capture-model"}},
            {"type": "session.closed", "usage": {"seconds": 2.5}, "transcript": "private transcript"},
        ]
    )
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, events, _AT, _AT)
        await meter.close()
        captured: Final = database.rows[-1].envelope
        assert "private transcript" not in captured
        result: Final = price_usage(captured)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(2.5 * 0.02)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("file_search", (False, True))
async def test_capture_replays_builtin_search_charge_without_retaining_results(file_search: bool) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.llms.openai import ResponsesAPIResponse
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger("aresponses")
    logger.set_request_pricing(
        "capture-model",
        "openai",
        {
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "search_context_cost_per_query": {"search_context_size_medium": 0.1},
            "file_search_cost_per_1k_calls": 0.025,
        },
    )
    logger.standard_built_in_tools_params = (
        {"file_search": {"type": "file_search", "vector_store_ids": ["store"]}}
        if file_search
        else {"web_search_options": {"search_context_size": "medium"}}
    )
    response: Final = ResponsesAPIResponse.model_validate(
        {
            "id": "search",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "capture-model",
            "output": [
                {"type": "file_search_call", "id": "search", "queries": ["private query"], "status": "completed"}
            ]
            if file_search
            else [
                {
                    "type": "web_search_call",
                    "id": "search",
                    "status": "completed",
                    "action": {"type": "search", "query": "private query"},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        }
    )
    source_cost: Final = logger.response_cost_calculator(response, ignore_stamped_cost=True)
    assert source_cost is not None and source_cost > 0.18
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        captured: Final = database.rows[-1].envelope
        assert "private query" not in captured
        result: Final = price_usage(captured)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(source_cost)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
async def test_capture_replays_multi_round_aggregate_with_one_fixed_request_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.responses.mcp.round_billing import billed_for_every_round, summed_cost_breakdown
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    monkeypatch.setattr(litellm, "cost_margin_config", {"global": {"fixed_amount": 0.1}})
    first_logger: Final = _capture_logger()
    final_logger: Final = _capture_logger()
    first: Final = ModelResponse(id="first", model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=4))
    final: Final = ModelResponse(id="final", model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=4))
    first._hidden_params["response_cost"] = first_logger.response_cost_calculator(first, ignore_stamped_cost=True)
    final._hidden_params["response_cost"] = final_logger.response_cost_calculator(final, ignore_stamped_cost=True)
    aggregate: Final = billed_for_every_round(first, first_logger.cost_breakdown, final, final_logger.cost_breakdown)
    breakdown: Final = summed_cost_breakdown(first, first_logger.cost_breakdown, final, final_logger.cost_breakdown)
    final_logger.cost_breakdown = breakdown
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, final_logger, aggregate, _AT, _AT)
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(2 * (10 * 0.01 + 4 * 0.02) + 0.1)
        assert float(result.total_cost) == pytest.approx(aggregate._hidden_params["response_cost"])
    finally:
        await meter.stop(1)


@pytest.mark.parametrize("reported", (-0.1, float("inf"), float("-inf"), float("nan"), True, False))
@pytest.mark.parametrize("has_rates", (False, True))
def test_invalid_upstream_price_cannot_become_an_authoritative_charge(reported: float, has_rates: bool) -> None:
    from pydantic import TypeAdapter

    envelope: Final = _retained_envelope(logging={"provider_reported_cost": reported})
    pricing: Final = TypeAdapter(dict[str, object]).validate_python(envelope["pricing"])
    if not has_rates:
        pricing["model_map_information"] = {}
    envelope["pricing"] = pricing
    result: Final = price_usage(json.dumps(envelope))
    if has_rates:
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(0.18)
        assert result.upstream_reported_cost is None
    else:
        assert result == Unpriced(usage_event_id="attempt", reason="pricing_inputs_missing")


def test_repricing_rejects_unretained_allowance_allocation() -> None:
    envelope: Final = _retained_envelope()
    envelope["finalized_cost"] = {"allowance_discount": 0.1}
    assert price_usage(json.dumps(envelope)) == Unpriced(
        usage_event_id="attempt", reason="allowance_allocation_missing"
    )


@pytest.mark.asyncio
async def test_retained_pricing_is_task_local_and_resets_on_cancellation() -> None:
    import asyncio

    from litellm.litellm_core_utils.llm_cost_calc.pricing_context import (
        RetainedPricing,
        pricing_discounts,
        retained_model_info,
        retained_models,
        retained_pricing,
    )

    outer: Final = RetainedPricing("{}", '{"openai":0.1}', "{}")
    inner: Final = RetainedPricing("{}", '{"openai":0.9}', "{}")
    ready: Final = asyncio.Event()
    hold: Final = asyncio.Event()
    restored: Final = asyncio.Event()

    async def child() -> None:
        try:
            with retained_pricing(inner):
                assert pricing_discounts()["openai"] == 0.9
                with pytest.raises(ValueError, match="no retained model"):
                    retained_model_info("gpt-4o-mini", "openai")
                ready.set()
                await hold.wait()
        finally:
            assert pricing_discounts()["openai"] == 0.1
            restored.set()

    assert retained_models() is None
    with retained_pricing(outer):
        task: Final = asyncio.create_task(child())
        try:
            await asyncio.wait_for(ready.wait(), timeout=5)
            assert pricing_discounts()["openai"] == 0.1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert restored.is_set()
        assert pricing_discounts()["openai"] == 0.1
    assert retained_models() is None
    assert retained_model_info("gpt-4o-mini", "openai") is None


def test_incomplete_retained_rates_remain_unpriced() -> None:
    result: Final = price_usage(
        json.dumps(_retained_envelope(rates={"input_cost_per_token": None, "output_cost_per_token": None}))
    )
    assert result == Unpriced(usage_event_id="attempt", reason="pricing_inputs_missing")


def test_explicit_zero_retained_token_rates_are_known_free() -> None:
    result: Final = price_usage(
        json.dumps(_retained_envelope(rates={"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}))
    )
    assert isinstance(result, PricingSnapshot)
    assert result.total_cost == Decimal("0")
    assert result.pricing_source == "cost_map"


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_tokens", [1, 4])
async def test_stream_replay_applies_adjustments_once_to_legacy_stamped_usage(
    monkeypatch: pytest.MonkeyPatch, completion_tokens: int
) -> None:
    import litellm
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    monkeypatch.setattr(litellm, "cost_discount_config", {"openai": 0.05})
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": {"percentage": 0.1, "fixed_amount": 0.001}})
    retained: Final = _capture_logger().request_pricing
    assert retained is not None
    monkeypatch.setitem(litellm.model_cost, "openai/capture-model", retained.models()["capture-model"])
    monkeypatch.setitem(litellm.model_cost, "capture-model", retained.models()["capture-model"])
    logger: Final = _logger(stream=True)
    response: Final = ModelResponse(
        id="terminal", model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=completion_tokens)
    )
    expected: Final = _logger(stream=True).response_cost_calculator(response, ignore_stamped_cost=True)
    assert expected is not None
    from litellm.litellm_core_utils.core_helpers import stamp_litellm_bill_on_usage

    stamp_litellm_bill_on_usage(response, response.usage, expected)
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        priced: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(priced, PricingSnapshot)
        assert float(priced.total_cost) == pytest.approx(expected)
        assert float(priced.original_cost or 0) == pytest.approx(0.1 + completion_tokens * 0.02)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [0.0, 0.123])
@pytest.mark.parametrize("mapped", [None, "mapped-cost-model", "openai/mapped-cost-model"])
@pytest.mark.parametrize("provider", ["openai", "litellm_proxy"])
async def test_capture_retains_provider_header_price_without_catalog(
    amount: float, mapped: str | None, provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    response: Final = ModelResponse(
        model="capture-model",
        usage=Usage(prompt_tokens=10, completion_tokens=4),
    )
    from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params

    set_response_cost_in_hidden_params(response, amount)
    import litellm

    logger: Final = _logger()
    logger.update_environment_variables(
        model="capture-model", custom_llm_provider=provider, litellm_params={}, optional_params={}
    )
    if mapped:
        monkeypatch.setitem(
            litellm.model_cost,
            "mapped-cost-model",
            {
                "mode": "chat",
                "litellm_provider": "openai",
                "input_cost_per_token": 0.01,
                "output_cost_per_token": 0.02,
            },
        )
        logger.model_call_details["litellm_params"]["metadata"] = {"model_info": {"base_model": mapped}}
    expected: Final = logger.response_cost_calculator(response, ignore_stamped_cost=True)
    expected_breakdown: Final = logger.cost_breakdown
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        if mapped:
            monkeypatch.setitem(litellm.model_cost, "mapped-cost-model", {"input_cost_per_token": 99.0})
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert result.total_cost == Decimal(str(amount))
        assert float(result.total_cost) == expected
        assert expected_breakdown is not None
        assert float(result.input_cost or 0) == pytest.approx(expected_breakdown["input_cost"])
        assert float(result.output_cost or 0) == pytest.approx(expected_breakdown["output_cost"])
        assert (result.input_cost or Decimal(0)) + (result.output_cost or Decimal(0)) == result.total_cost
        assert result.pricing_source == "upstream_header"
    finally:
        await meter.stop(1)


@pytest.mark.parametrize("metadata", [{"input_cost_per_token": None}, {"service_tier": "default"}])
def test_catalog_provenance_ignores_metadata_without_price_overrides(metadata: dict[str, object]) -> None:
    result: Final = price_usage(json.dumps(_retained_envelope(deployment=metadata, request=metadata)))
    assert isinstance(result, PricingSnapshot)
    assert result.pricing_source == "cost_map"


@pytest.mark.parametrize("call_type", ["get_responses", "aget_responses"])
def test_terminal_response_replay_prices_usage_without_changing_event_identity(call_type: str) -> None:
    from pydantic import TypeAdapter

    original: Final = _retained_envelope()
    event: Final = TypeAdapter(dict[str, object]).validate_python(original["event"])
    result: Final = price_usage(json.dumps({**original, "event": {**event, "call_type": call_type}}))
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx(10 * 0.01 + 4 * 0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("freeze_pricing", [False, True])
async def test_batch_terminal_capture_refreshes_row_prices_after_creation(
    monkeypatch: pytest.MonkeyPatch, freeze_pricing: bool
) -> None:
    import litellm
    from litellm.proxy.spend_tracking.usage_capture import capture_batch_callback, capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    rates: Final = {
        "mode": "chat",
        "litellm_provider": "openai",
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.02,
        "input_cost_per_token_batches": 0.003,
        "output_cost_per_token_batches": 0.004,
    }
    monkeypatch.setitem(litellm.model_cost, "capture-model", rates)
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": {"fixed_amount": 0.1}})
    monkeypatch.setitem(litellm.model_cost, "second-model", {**rates, "input_cost_per_token_batches": 0.005})
    logger: Final = _logger("aretrieve_batch")
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    usage: Final = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    response: Final = {
        "id": "batch",
        "status": "completed",
        "usage": {
            **usage,
            "batch_usage_by_model": {"capture-model": usage, "second-model": usage},
            "batch_usage_by_request": [
                {"model": "capture-model", "usage": usage},
                {"model": "capture-model", "usage": usage},
                {"model": "second-model", "usage": usage},
            ],
        },
    }
    try:
        capture_logging_usage(meter, logger, {"id": "batch", "status": "in_progress"}, _AT, _AT, freeze_only=True)
        if freeze_pricing:
            capture_logging_usage(meter, logger, response, _AT, _AT)
        await capture_batch_callback(
            meter, database.load, {**logger.model_call_details, "litellm_logging_obj": logger}, response, _AT, _AT
        )
        first_rows: Final = database.rows
        monkeypatch.setitem(litellm.model_cost, "capture-model", {**rates, "input_cost_per_token_batches": 99.0})
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await capture_batch_callback(meter, database.load, logger.model_call_details, response, _AT, _AT)
        await meter.close()
        assert database.rows == first_rows
        results: Final = tuple(price_usage(row.envelope) for row in database.rows)
        priced: Final = tuple(result for result in results if isinstance(result, PricingSnapshot))
        assert len(priced) == 3
        assert sorted(float(price.total_cost) for price in priced) == pytest.approx([0.046, 0.046, 0.066])
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["get_responses", "aget_responses"])
async def test_background_capture_compiles_terminal_usage_once(call_type: str) -> None:
    from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger(call_type)
    logger.update_environment_variables(
        model="capture-model",
        custom_llm_provider="openai",
        optional_params={},
        litellm_params={"metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN}},
    )
    response: Final = ResponsesAPIResponse.model_validate(
        {
            "id": "background-job",
            "object": "response",
            "created_at": int(_AT.timestamp()),
            "status": "completed",
            "background": True,
            "model": "capture-model",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        }
    )
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        assert len(database.rows) == 1
        assert database.rows[0].event.call_type.value == call_type
        assert database.rows[0].event.event_id == "background-job_background_cost"
        result: Final = price_usage(database.rows[0].envelope)
        assert isinstance(result, PricingSnapshot)
        assert float(result.total_cost) == pytest.approx(10 * 0.01 + 4 * 0.02)
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["call_mcp_tool", "asend_message"])
async def test_failed_fixed_call_without_usage_does_not_invent_a_free_charge(call_type: str) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_callback
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _logger(call_type)
    logger.model_call_details["response_cost"] = 0.0
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_callback(
            meter,
            {**logger.model_call_details, "litellm_logging_obj": logger, "log_event_type": "failed_api_call"},
            RuntimeError("Request rejected before dispatch"),
            _AT,
            _AT,
        )
        await meter.close()
        outbound: Final = database.rows[-1]
        assert outbound.event.outcome is Outcome.FAILED
        assert price_usage(outbound.envelope) == Unpriced(
            usage_event_id=outbound.event.event_id, reason="usage_unknown"
        )
    finally:
        await meter.stop(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("billable", [False, True])
async def test_capture_preserves_completed_guardrail_fee_without_header_finalization(
    stream: bool, billable: bool
) -> None:
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import ModelResponse, Usage
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _capture_logger(stream=stream)
    logger.model_call_details["litellm_params"]["metadata"] = {
        "standard_logging_guardrail_information": [
            {"guardrail_name": "late-fee", "guardrail_cost": 0.03, "guardrail_cost_in_spend": billable}
        ]
    }
    response: Final = ModelResponse(
        id="guarded", model="capture-model", usage=Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    )
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.close()
        result: Final = price_usage(database.rows[-1].envelope)
        assert isinstance(result, PricingSnapshot)
        assert result.guardrail_cost == Decimal("0.03" if billable else "0")
        assert float(result.total_cost) == pytest.approx(0.18 + (0.03 if billable else 0))
    finally:
        await meter.stop(1)


@pytest.mark.parametrize("mode", [None, "chat"])
def test_header_split_retains_token_rates_when_catalog_mode_is_unspecified(mode: str | None) -> None:
    result: Final = price_usage(
        json.dumps(_retained_envelope(rates={"mode": mode}, logging={"provider_reported_cost": 0.18}))
    )
    assert isinstance(result, PricingSnapshot)
    assert float(result.total_cost) == pytest.approx(0.18)
    assert float(result.input_cost or 0) == pytest.approx(0.1)
    assert float(result.output_cost or 0) == pytest.approx(0.08)


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_first", (False, True))
async def test_batch_logging_and_rows_have_one_charge_owner(
    monkeypatch: pytest.MonkeyPatch, logging_first: bool
) -> None:
    import litellm
    from litellm.batches.batch_utils import batch_cost_and_usage
    from litellm.proxy.spend_tracking.usage_capture import (
        UsageBatchWriter,
        capture_batch_callback,
        capture_callback,
        capture_logging_usage,
    )
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.utils import LiteLLMBatch
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    rates: Final = {
        "input_cost_per_token_batches": 0.003,
        "output_cost_per_token_batches": 0.004,
        "input_cost_per_token": 0.006,
        "output_cost_per_token": 0.008,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, "capture-model", rates)
    billed: Final = batch_cost_and_usage(
        [
            {
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "capture-model",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                }
            }
            for _ in range(2)
        ],
        "openai",
    )
    response: Final = LiteLLMBatch(
        id="owner-job",
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id="file",
        completion_window="24h",
        status="completed",
        created_at=1,
        completed_at=2,
        usage=billed.usage,
    )
    database: Final = StoredUsage()
    writer: Final = UsageBatchWriter(database.insert, database.load)
    meter: Final = DurableUsageMeter(writer.write, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    logger: Final = _logger("aretrieve_batch")
    capture_logging_usage(meter, logger, response, _AT, _AT, freeze_only=True)
    try:
        for _ in range(2):
            if logging_first:
                capture_callback(
                    meter,
                    {**logger.model_call_details, "litellm_logging_obj": logger},
                    {
                        **response.model_dump(),
                        "id": "provider-job",
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                    },
                    _AT,
                    _AT,
                )
                capture_logging_usage(meter, logger, response, _AT, _AT)
            await capture_batch_callback(
                meter, database.load, {**logger.model_call_details, "litellm_logging_obj": logger}, response, _AT, _AT
            )
            if not logging_first:
                capture_callback(
                    meter,
                    {**logger.model_call_details, "litellm_logging_obj": logger},
                    {
                        **response.model_dump(),
                        "id": "provider-job",
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                    },
                    _AT,
                    _AT,
                )
                capture_logging_usage(meter, logger, response, _AT, _AT)
            await meter.flush()
        costs: Final = tuple(
            price_usage(row.envelope) for row in database.rows if isinstance(row.event, OutboundUsageEvent)
        )
        priced: Final = tuple(cost for cost in costs if isinstance(cost, PricingSnapshot))
        assert len(priced) == 2, costs
        assert sum(float(cost.total_cost) for cost in priced) == pytest.approx(2 * (10 * 0.003 + 5 * 0.004))
    finally:
        await meter.stop(1)


@pytest.mark.parametrize("characters", (0, 17))
def test_speech_replay_prices_retained_characters(characters: int) -> None:
    original: Final = _retained_envelope(
        rates={"input_cost_per_character": 0.003, "mode": "audio_speech"}, logging={"prompt_characters": characters}
    )
    event: Final = OutboundUsageEvent.model_validate(original["event"]).model_copy(
        update={"call_type": CallTypes.aspeech}
    )
    result: Final = price_usage(json.dumps({**original, "event": event.model_dump(mode="json")}))
    assert isinstance(result, PricingSnapshot), result
    assert float(result.total_cost) == pytest.approx(characters * 0.003)


def test_batch_ocr_replay_prices_pages_and_annotations() -> None:
    original: Final = _retained_envelope(
        rates={"ocr_cost_per_page_batches": 0.003, "annotation_cost_per_page_batches": 0.004, "mode": "ocr"},
        usage={"pages_processed": 3, "pages_processed_annotation": 2},
    )
    event: Final = OutboundUsageEvent.model_validate(original["event"]).model_copy(
        update={"call_type": CallTypes.aretrieve_batch}
    )
    result: Final = price_usage(
        json.dumps(
            {
                **original,
                "event": {**event.model_dump(mode="json"), "event_id": "batch:row:0"},
                "parent_request_id": "batch",
            }
        )
    )
    assert isinstance(result, PricingSnapshot), result
    assert float(result.total_cost) == pytest.approx(3 * 0.003 + 2 * 0.004)


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_characters", (None, 5))
async def test_speech_capture_replays_billed_quantities_without_input_text(reported_characters: int | None) -> None:
    from httpx import Response

    from litellm.litellm_core_utils.hidden_params import set_hidden_param
    from litellm.proxy.spend_tracking.usage_capture import capture_logging_usage
    from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureSettings
    from litellm.types.llms.openai import HttpxBinaryResponseContent
    from tests.unit.proxy.spend_tracking.test_usage_capture import StoredUsage, _logger
    from tests.unit.proxy.spend_tracking.test_usage_meter import _manual_flush

    logger: Final = _logger("aspeech")
    logger.set_request_pricing("capture-model", "openai", {"input_cost_per_character": 0.003, "mode": "audio_speech"})
    logger.model_call_details["input"] = "privatespeechinput"
    response: Final = HttpxBinaryResponseContent(Response(200, content=b"audio"))
    if reported_characters is not None:
        set_hidden_param(response, "billed_characters", reported_characters)
    expected: Final = (reported_characters if reported_characters is not None else len("privatespeechinput")) * 0.003
    assert logger.response_cost_calculator(response) == pytest.approx(expected)
    database: Final = StoredUsage()
    meter: Final = DurableUsageMeter(database.insert, UsageCaptureSettings(), wait_for_batch=_manual_flush)
    try:
        capture_logging_usage(meter, logger, response, _AT, _AT)
        await meter.flush()
        envelope: Final = next(row.envelope for row in database.rows if isinstance(row.event, OutboundUsageEvent))
        assert "privatespeechinput" not in envelope
        priced: Final = price_usage(envelope)
        assert isinstance(priced, PricingSnapshot), priced
        assert float(priced.total_cost) == pytest.approx(expected)
    finally:
        await meter.stop(1)


@pytest.mark.parametrize("characters", (None, -1, True))
def test_character_pricing_does_not_invent_zero_for_missing_or_invalid_quantity(characters: object) -> None:
    original: Final = _retained_envelope(
        rates={"input_cost_per_character": 0.003}, logging={"prompt_characters": characters}
    )
    event: Final = OutboundUsageEvent.model_validate(original["event"]).model_copy(
        update={"call_type": CallTypes.aspeech}
    )
    assert price_usage(json.dumps({**original, "event": event.model_dump(mode="json")})) == Unpriced(
        usage_event_id=event.event_id, reason="usage_unknown"
    )
