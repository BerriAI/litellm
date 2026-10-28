import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from litellm.accounting import (
    CostEvent,
    DimensionName,
    MetricName,
    Outcome,
    SpendLogAdapterError,
    SpendLogContext,
    SpendLogEvents,
    Unpriced,
    adapt_spend_log,
    measurements_for,
)
from litellm.proxy._types import SpendLogsPayload
from litellm.types.utils import CallTypes

_START: Final = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
_END: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)


def _row() -> SpendLogsPayload:
    return SpendLogsPayload(
        request_id="provider-response",
        call_type="acompletion",
        api_key="hashed-key",
        spend=0.125,
        total_tokens=12,
        prompt_tokens=10,
        completion_tokens=2,
        startTime=_START,
        endTime=_END,
        completionStartTime=None,
        model="synthetic-model",
        model_id="deployment",
        model_group="alias",
        mcp_namespaced_tool_name=None,
        agent_id=None,
        api_base="",
        user="user",
        metadata=json.dumps({"usage_object": {"prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 1}}}),
        cache_hit="False",
        cache_key="",
        request_tags='["alpha", "beta", "alpha"]',
        team_id="team",
        organization_id="org",
        end_user="customer-id",
        requester_ip_address=None,
        custom_llm_provider="synthetic-provider",
        messages="private prompt",
        response="private response",
        proxy_server_request="private body",
        session_id=None,
        request_duration_ms=1000,
        status="success",
        litellm_call_id="call-id",
    )


def test_success_preserves_attribution_charge_and_complete_measurement_sets():
    result = adapt_spend_log(_row(), context=SpendLogContext(usage_known=True, price_known=True, request_id="caller"))
    assert isinstance(result, SpendLogEvents)
    assert result.inbound is not None
    assert result.inbound.event_id == "inbound:caller"
    assert result.outbound.event_id == "outbound:provider-response"
    assert result.outbound.attempt_id == "provider-response"
    assert result.outbound.retry_of_attempt_id is None
    assert result.outbound.request_id == "caller"
    assert result.outbound.outcome is Outcome.SUCCEEDED
    assert result.outbound.call_type is CallTypes.acompletion
    assert isinstance(result.cost, CostEvent)
    assert result.cost.event_id == "cost:provider-response"
    assert result.cost.usage_event_id == result.outbound.event_id
    assert result.cost.charges[0].value == Decimal("0.125")
    assert result.cost.charges[0].unit == "USD"
    assert result.cost.occurred_at == _START
    assert {(d.name, d.value) for d in result.outbound.dimensions} == {
        (DimensionName.KEY, "hashed-key"),
        (DimensionName.USER, "user"),
        (DimensionName.TEAM, "team"),
        (DimensionName.ORGANIZATION, "org"),
        (DimensionName.END_USER, "customer-id"),
        (DimensionName.MODEL, "synthetic-model"),
        (DimensionName.MODEL_GROUP, "alias"),
        (DimensionName.DEPLOYMENT, "deployment"),
        (DimensionName.PROVIDER, "synthetic-provider"),
        (DimensionName.TAG, "alpha"),
        (DimensionName.TAG, "beta"),
    }
    assert {m.metric.name: m.metric.value for m in measurements_for(result.inbound)} == {
        MetricName.REQUESTS: Decimal(1),
        MetricName.SUCCESSFUL_REQUESTS: Decimal(1),
    }
    usage = measurements_for(result.outbound)
    assert {m.metric.name: m.metric.value for m in usage} == {
        MetricName.PROMPT_TOKENS: Decimal(10),
        MetricName.COMPLETION_TOKENS: Decimal(2),
        MetricName.TOTAL_TOKENS: Decimal(12),
        MetricName.CACHE_READ_INPUT_TOKENS: Decimal(3),
        MetricName.CACHE_CREATION_INPUT_TOKENS: Decimal(1),
    }
    assert all(m.source_event_id == result.outbound.event_id and m.occurred_at == _START for m in usage)
    assert measurements_for(result.cost)[0].metric == result.cost.charges[0]
    assert measurements_for(result.cost)[0].source_event_id == result.cost.event_id
    assert "private" not in result.outbound.model_dump_json()


@pytest.mark.parametrize(
    ("row_id", "metadata", "call_type"),
    [
        ("provider-response-attempt-1", {"billed_retry_attempt": True, "attempted_retries": 7}, "acompletion"),
        ("batch_batch_cost", {}, "aretrieve_batch"),
        ("response_background_cost", {}, "acompletion"),
        ("classifier", {"internal_call_origin": "auto_router"}, "acompletion"),
    ],
)
def test_internal_retry_and_settlement_rows_add_usage_and_cost_without_caller_counts(row_id, metadata, call_type):
    row = {**_row(), "request_id": row_id, "metadata": json.dumps(metadata), "call_type": call_type}
    result = adapt_spend_log(row, context=SpendLogContext(usage_known=True, price_known=True, request_id="caller"))
    assert isinstance(result, SpendLogEvents)
    assert result.inbound is None
    assert result.outbound.attempt_id == row_id
    assert result.outbound.event_id == f"outbound:{row_id}"
    assert result.outbound.retry_of_attempt_id is None
    assert result.outbound.request_id == "caller"
    assert isinstance(result.cost, CostEvent)
    assert result.cost.charges[0].value == Decimal("0.125")
    assert measurements_for(result.outbound)[0].metric.value == Decimal(10)
    assert all(m.metric.name is not MetricName.REQUESTS for m in measurements_for(result.outbound))


def test_explicit_retry_parent_is_preserved_without_inventing_caller_identity():
    result = adapt_spend_log(
        _row(), context=SpendLogContext(usage_known=True, price_known=True, retry_of_attempt_id="earlier-attempt")
    )
    assert isinstance(result, SpendLogEvents)
    assert result.inbound is None
    assert result.outbound.request_id is None
    assert result.outbound.retry_of_attempt_id == "earlier-attempt"


@pytest.mark.parametrize(
    ("status", "override", "expected"),
    [
        ("failure", None, Outcome.FAILED),
        ("failure", Outcome.CANCELLED, Outcome.CANCELLED),
    ],
)
def test_failed_and_cancelled_calls_keep_incurred_charges(status, override, expected):
    row = {**_row(), "status": status}
    result = adapt_spend_log(
        row, context=SpendLogContext(usage_known=True, price_known=True, request_id="caller", outcome=override)
    )
    assert isinstance(result, SpendLogEvents)
    assert result.outbound.outcome is expected
    assert result.inbound is not None
    assert result.inbound.outcome is expected
    assert measurements_for(result.inbound)[1].metric.name is MetricName.FAILED_REQUESTS
    assert isinstance(result.cost, CostEvent)
    assert result.cost.charges[0].value == Decimal("0.125")


@pytest.mark.parametrize("price_known", [True, False])
def test_known_free_cache_hits_remain_distinct_from_unknown_price(price_known):
    row = {**_row(), "request_id": "cached_cache_hit123", "cache_hit": "True", "spend": 0.0}
    context = SpendLogContext(usage_known=True, price_known=price_known, request_id="caller")
    result = adapt_spend_log(row, context=context)
    assert isinstance(result, SpendLogEvents)
    assert result.outbound.attempt_id == "cached_cache_hit123"
    assert result.inbound is not None
    assert result == adapt_spend_log(row, context=context)
    if price_known:
        assert isinstance(result.cost, CostEvent)
        assert result.cost.charges[0].value == Decimal(0)
    else:
        assert isinstance(result.cost, Unpriced)
        assert result.cost.usage_event_id == result.outbound.event_id
        assert result.cost.reason == "No authoritative price supplied"


def test_unknown_usage_is_not_zero_and_does_not_suppress_caller_activity():
    result = adapt_spend_log(_row(), context=SpendLogContext(usage_known=False, price_known=False, request_id="caller"))
    assert isinstance(result, SpendLogEvents)
    assert result.outbound.metrics is None
    assert measurements_for(result.outbound) == ()
    assert result.inbound is not None
    assert measurements_for(result.inbound)[0].metric.value == Decimal(1)
    assert isinstance(result.cost, Unpriced)


def test_missing_price_stays_unpriced_even_if_context_claims_known():
    row = {**_row(), "spend": None}
    result = adapt_spend_log(row, context=SpendLogContext(usage_known=True, price_known=True))
    assert isinstance(result, SpendLogEvents)
    assert isinstance(result.cost, Unpriced)


def test_mcp_tool_attribution_is_preserved_without_a_model():
    row = {
        **_row(),
        "model": "",
        "call_type": "call_mcp_tool",
        "mcp_namespaced_tool_name": "tools:lookup",
        "agent_id": "agent",
    }
    result = adapt_spend_log(row, context=SpendLogContext(usage_known=True, price_known=True, request_id="caller"))
    assert isinstance(result, SpendLogEvents)
    assert (DimensionName.TOOL, "tools:lookup") in {(d.name, d.value) for d in result.outbound.dimensions}
    assert (DimensionName.AGENT, "agent") in {(d.name, d.value) for d in result.outbound.dimensions}
    assert not any(d.name is DimensionName.MODEL for d in result.outbound.dimensions)
    assert isinstance(result.cost, CostEvent)
    assert result.cost.charges[0].value == Decimal("0.125")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata", "invalid-json"),
        ("call_type", "unknown-call"),
        ("prompt_tokens", -1),
        ("spend", float("nan")),
        ("spend", -1.0),
        ("startTime", "2026-01-01T00:00:00"),
        ("endTime", "2025-01-01T00:00:00Z"),
        ("request_tags", '[" "]'),
    ],
)
def test_invalid_rows_return_a_sanitized_error(field, value):
    row = {**_row(), field: value}
    result = adapt_spend_log(row, context=SpendLogContext(usage_known=True, price_known=True))
    assert result == SpendLogAdapterError(reason="Invalid spend-log accounting fields")


def test_billed_retry_and_success_reconcile_to_one_request_and_both_charges():
    retry_row = {**_row(), "request_id": "attempt-1", "status": "failure", "metadata": '{"billed_retry_attempt":true}'}
    retry = adapt_spend_log(retry_row, context=SpendLogContext(usage_known=True, price_known=True, request_id="caller"))
    success = adapt_spend_log(
        _row(),
        context=SpendLogContext(
            usage_known=True, price_known=True, request_id="caller", retry_of_attempt_id="attempt-1"
        ),
    )
    assert isinstance(retry, SpendLogEvents) and isinstance(success, SpendLogEvents)
    assert retry.inbound is None and success.inbound is not None
    assert isinstance(retry.cost, CostEvent) and isinstance(success.cost, CostEvent)
    measurements = (
        *measurements_for(retry.outbound),
        *measurements_for(retry.cost),
        *measurements_for(success.inbound),
        *measurements_for(success.outbound),
        *measurements_for(success.cost),
    )
    assert sum(m.metric.value for m in measurements if m.metric.name is MetricName.REQUESTS) == Decimal(1)
    assert sum(m.metric.value for m in measurements if m.metric.name is MetricName.TOTAL_TOKENS) == Decimal(24)
    assert sum(m.metric.value for m in measurements if m.metric.name is MetricName.SPEND) == Decimal("0.250")
    assert success.outbound.retry_of_attempt_id == retry.outbound.attempt_id
    assert retry.cost.event_id != success.cost.event_id


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
def test_nonfinite_cache_usage_returns_an_error_instead_of_escaping_the_adapter(value):
    row = {**_row(), "metadata": '{"usage_object":{"cache_read_input_tokens":' + value + "}}"}
    result = adapt_spend_log(row, context=SpendLogContext(usage_known=True, price_known=True))
    assert result == SpendLogAdapterError(reason="Invalid spend-log accounting fields")
