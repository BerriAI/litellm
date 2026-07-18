"""Tests for the V2 OTEL GenAI client metrics.

Drives the real success path: the six ``gen_ai.client.*`` histograms are emitted
through ``OpenTelemetryV2.async_log_success_event`` into an injected
``InMemoryMetricReader``, and attributes/values are read straight off the
recorded data points (``resource_metrics`` -> ``scope_metrics`` -> ``metrics`` ->
``data.data_points``). The cardinality filter is resolved lazily from
``litellm.callback_settings['otel']['attributes']``, which the proxy populates
after the logger is built, so those tests set it AFTER construction. A
misconfigured filter (``gen_ai.token.type`` in a list, include+exclude together)
raises out of ``GenAIMetricRecorder.record`` -- asserted directly at the recorder
layer -- and the logger turns that raise into a single ERROR ("metrics disabled")
plus a quiet no-op for the rest of the process, asserted at the logger layer so
the misconfig never breaks a request nor spams a log line per request.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

import litellm  # noqa: E402
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.model.config import (  # noqa: E402
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.plumbing.metrics import (  # noqa: E402
    GenAIMetricRecorder,
    create_genai_metrics,
)
from litellm.integrations.otel.plumbing.providers import (  # noqa: E402
    resolve_meter_provider,
)

OPERATION_DURATION = "gen_ai.client.operation.duration"
TOKEN_USAGE = "gen_ai.client.token.usage"
TOKEN_COST = "gen_ai.client.token.cost"
TIME_TO_FIRST_TOKEN = "gen_ai.client.response.time_to_first_token"
TIME_PER_OUTPUT_TOKEN = "gen_ai.client.response.time_per_output_token"
RESPONSE_DURATION = "gen_ai.client.response.duration"

ALL_METRICS = frozenset(
    {
        OPERATION_DURATION,
        TOKEN_USAGE,
        TOKEN_COST,
        TIME_TO_FIRST_TOKEN,
        TIME_PER_OUTPUT_TOKEN,
        RESPONSE_DURATION,
    }
)

TOKEN_TYPE = "gen_ai.token.type"
MODEL_KEY = "gen_ai.request.model"

# Each is a member of VALID_METRIC_ATTRIBUTE_NAMES and is stamped on the metric
# by default (proven by the no-filter test below).
HIGH_CARDINALITY_KEYS = (
    "hidden_params",
    "metadata.user_api_key_hash",
    "metadata.requester_ip_address",
    "metadata.requester_metadata",
    "metadata.applied_guardrails",
)

PROMPT_TOKENS = 137
COMPLETION_TOKENS = 89
RESPONSE_COST = 0.0023


def _build_call(stream: bool = True):
    """A captured success-call (kwargs, response_obj, start, end) that exercises
    every one of the six metrics: usage for token.usage, response_cost for cost,
    streaming + timing for the response-time histograms."""
    start = datetime(2026, 6, 12, 12, 0, 0)
    api_call_start = start + timedelta(seconds=0.1)
    completion_start = start + timedelta(seconds=0.5)
    end = start + timedelta(seconds=1.0)
    kwargs = {
        "model": "gpt-4o-mini",
        "call_type": "completion",
        "litellm_params": {"custom_llm_provider": "openai"},
        "optional_params": {"stream": stream},
        "response_cost": RESPONSE_COST,
        "api_call_start_time": api_call_start,
        "completion_start_time": completion_start,
        "end_time": end,
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": "hash-abc123",
                "requester_ip_address": "10.0.0.7",
                "requester_metadata": {"team": "alpha", "tier": "gold"},
                "applied_guardrails": ["pii", "toxicity"],
            },
            "hidden_params": {"litellm_call_id": "abc", "model_id": "m-1"},
        },
    }
    response_obj = {
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
        }
    }
    return kwargs, response_obj, start, end


def _logger(reader, *, enable_metrics: bool):
    return OpenTelemetryV2(
        config=OpenTelemetryV2Config(
            exporter="in_memory", enable_metrics=enable_metrics
        ),
        meter_provider=MeterProvider(metric_readers=[reader]),
    )


def _metrics_by_name(reader):
    """{metric_name: [data_point, ...]} from everything the reader has collected."""
    data = reader.get_metrics_data()
    out: dict = {}
    if not data or not getattr(data, "resource_metrics", None):
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out.setdefault(m.name, []).extend(m.data.data_points)
    return out


def _drive_success(reader, callback_settings_attributes=None):
    """Construct a metrics-on logger, optionally populate callback_settings AFTER
    construction (mirroring the proxy ordering), run the real success hook."""
    logger = _logger(reader, enable_metrics=True)
    previous = litellm.callback_settings
    if callback_settings_attributes is not None:
        litellm.callback_settings = {
            "otel": {"attributes": callback_settings_attributes}
        }
    try:
        kwargs, response_obj, start, end = _build_call()
        asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))
    finally:
        litellm.callback_settings = previous
    return _metrics_by_name(reader)


def test_all_six_metrics_emitted_when_enabled():
    """A successful streaming call with metrics on emits exactly the six
    gen_ai.client.* histograms, and token.usage splits into an input and an
    output point carrying the right token counts."""
    metrics = _drive_success(InMemoryMetricReader())

    assert set(metrics.keys()) == set(ALL_METRICS)

    token_points = metrics[TOKEN_USAGE]
    by_type = {dp.attributes[TOKEN_TYPE]: dp for dp in token_points}
    assert set(by_type) == {"input", "output"}
    assert by_type["input"].sum == PROMPT_TOKENS
    assert by_type["output"].sum == COMPLETION_TOKENS

    cost_points = metrics[TOKEN_COST]
    assert len(cost_points) == 1
    assert cost_points[0].sum == pytest.approx(RESPONSE_COST)


def test_time_to_first_token_is_streaming_only():
    """time_to_first_token is gated on streaming: a non-streaming call emits the
    other five metrics but never that one."""
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    kwargs, response_obj, start, end = _build_call(stream=False)
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    names = set(_metrics_by_name(reader).keys())
    assert TIME_TO_FIRST_TOKEN not in names
    assert names == set(ALL_METRICS) - {TIME_TO_FIRST_TOKEN}


def test_metrics_disabled_records_nothing():
    """enable_metrics=False: the recorder is never built, so the injected reader
    sees no gen_ai.client.* series even though the success hook runs."""
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=False)
    kwargs, response_obj, start, end = _build_call()
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    assert set(_metrics_by_name(reader).keys()).isdisjoint(ALL_METRICS)


def test_metrics_off_by_default_records_nothing():
    """The default config has metrics off, so a default logger records nothing."""
    reader = InMemoryMetricReader()
    logger = OpenTelemetryV2(
        config=OpenTelemetryV2Config(exporter="in_memory"),
        meter_provider=MeterProvider(metric_readers=[reader]),
    )
    kwargs, response_obj, start, end = _build_call()
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    assert set(_metrics_by_name(reader).keys()).isdisjoint(ALL_METRICS)


def test_exclude_list_strips_high_cardinality_across_metrics():
    """exclude_list set AFTER construction (the proxy path) removes every
    high-cardinality key from more than one metric while the low-cardinality
    model attribute survives."""
    metrics = _drive_success(
        InMemoryMetricReader(),
        callback_settings_attributes={"exclude_list": list(HIGH_CARDINALITY_KEYS)},
    )
    excluded = set(HIGH_CARDINALITY_KEYS)

    for name in (OPERATION_DURATION, TOKEN_USAGE):
        points = metrics[name]
        assert points, f"{name} was not recorded"
        for dp in points:
            keys = set(dp.attributes.keys())
            assert excluded.isdisjoint(keys), f"{name} leaked {excluded & keys}"
            assert MODEL_KEY in keys


def test_include_list_allows_only_listed_attributes():
    """include_list caps emitted attributes to exactly the listed set;
    gen_ai.token.type is the only key permitted beyond it, and only on the
    token-usage metric."""
    include = [MODEL_KEY, "gen_ai.system"]
    metrics = _drive_success(
        InMemoryMetricReader(),
        callback_settings_attributes={"include_list": include},
    )
    allowed = set(include)

    for dp in metrics[OPERATION_DURATION]:
        assert set(dp.attributes.keys()) == allowed

    for dp in metrics[TOKEN_USAGE]:
        assert set(dp.attributes.keys()) - {TOKEN_TYPE} == allowed


def test_no_filter_keeps_high_cardinality_keys():
    """Backward compatibility: without an attributes config every high-cardinality
    key the call carries is still stamped, so the filter tests above prove a real
    removal rather than a key that was never present."""
    metrics = _drive_success(InMemoryMetricReader())
    expected = set(HIGH_CARDINALITY_KEYS)

    for name in (OPERATION_DURATION, TOKEN_USAGE):
        for dp in metrics[name]:
            assert expected.issubset(set(dp.attributes.keys()))


def test_metrics_reach_operator_configured_global_provider(monkeypatch):
    """Regression: with no meter provider injected, the six gen_ai.client.*
    histograms must record through the operator's globally configured
    MeterProvider so its readers/exporters receive them. Before the fix the logger
    built an isolated provider and the operator's reader saw nothing."""
    from opentelemetry import metrics

    reader = InMemoryMetricReader()
    operator_provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: operator_provider)

    logger = OpenTelemetryV2(
        config=OpenTelemetryV2Config(exporter="in_memory", enable_metrics=True),
    )
    kwargs, response_obj, start, end = _build_call()
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    assert set(_metrics_by_name(reader).keys()) == set(ALL_METRICS)
    operator_provider.shutdown()


def test_resolve_meter_provider_prefers_injected():
    """An injected provider is used verbatim, never replaced by the global."""
    injected = MeterProvider(metric_readers=[InMemoryMetricReader()])
    resolved = resolve_meter_provider(
        OpenTelemetryV2Config(exporter="in_memory"), injected
    )
    assert resolved is injected
    injected.shutdown()


def test_resolve_meter_provider_honors_operator_noop(monkeypatch):
    """An operator that disabled metrics with a NoOpMeterProvider is not silently
    overridden by a freshly built provider."""
    from opentelemetry import metrics
    from opentelemetry.metrics import NoOpMeterProvider

    noop = NoOpMeterProvider()
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: noop)

    resolved = resolve_meter_provider(OpenTelemetryV2Config(exporter="in_memory"))
    assert resolved is noop


def _recorder(monkeypatch, attributes):
    """A recorder wired to a fresh in-memory meter, with callback_settings carrying
    `attributes`. record() resolves the filter lazily from there, so a misconfig
    raises out of record() at this layer (the logger turns it into log-once)."""
    monkeypatch.setattr(
        litellm,
        "callback_settings",
        {"otel": {"attributes": attributes}},
        raising=False,
    )
    meter = MeterProvider(metric_readers=[InMemoryMetricReader()]).get_meter("test")
    return GenAIMetricRecorder(create_genai_metrics(meter), callback_name=None)


@pytest.mark.parametrize(
    "attributes",
    [
        {"exclude_list": [TOKEN_TYPE]},
        {"include_list": [TOKEN_TYPE]},
    ],
)
def test_token_type_rejected_from_either_list(attributes, monkeypatch):
    """gen_ai.token.type is a structural discriminator stamped onto the
    input/output series after filtering; it cannot itself be filtered without
    collapsing the two series. Listing it in either list is rejected by the
    recorder rather than silently ignored, so the misconfig is caught at all."""
    recorder = _recorder(monkeypatch, attributes)
    kwargs, response_obj, start, end = _build_call()
    with pytest.raises(ValueError) as exc_info:
        recorder.record(kwargs, response_obj, start, end)
    # The dedicated discriminator guard, not the generic unknown-name path: assert
    # the specific reason so dropping that guard (and falling through to "unknown
    # attribute name") is caught.
    assert "discriminator" in str(exc_info.value)
