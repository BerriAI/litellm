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

The failure path is driven the same way, through the real
``OpenTelemetryV2.async_log_failure_event``: a failed call records
``gen_ai.client.operation.duration`` and nothing else, tagged with ``error.type``,
and a success driven through the same reader keeps a datapoint whose attributes are
byte-for-byte what it had before the failure path existed -- the guard for every
dashboard already querying that histogram.
"""

import asyncio
import json
from datetime import datetime, timedelta

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

import litellm  # noqa: E402
from litellm.constants import (  # noqa: E402
    LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL,
)
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
TOKEN_COST = "gen_ai.usage.cost"
TIME_TO_FIRST_TOKEN = "gen_ai.server.time_to_first_token"
TIME_PER_OUTPUT_TOKEN = "gen_ai.server.time_per_output_token"
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
OPERATION_KEY = "gen_ai.operation.name"
PROVIDER_NAME_KEY = "gen_ai.provider.name"
SYSTEM_KEY = "gen_ai.system"

# Keys inside the ceiling that an operator's filter must still be able to remove.
# Every one is bounded, so it survives the ceiling and only the operator's own
# exclude_list takes it off; that is what makes the filter tests non-vacuous.
FILTERABLE_KEYS = (
    "hidden_params",
    "metadata.user_api_key_hash",
    "metadata.user_api_key_team_id",
)

PROMPT_TOKENS = 137
COMPLETION_TOKENS = 89
RESPONSE_COST = 0.0023


def _build_call(
    stream: bool = True,
    provider: str | None = "openai",
    call_type: str = "completion",
):
    """A captured success-call (kwargs, response_obj, start, end) that exercises
    every one of the six metrics: usage for token.usage, response_cost for cost,
    streaming + timing for the response-time histograms.

    ``provider=None`` omits ``custom_llm_provider`` entirely, reproducing a call
    litellm could not attribute to a provider."""
    start = datetime(2026, 6, 12, 12, 0, 0)
    api_call_start = start + timedelta(seconds=0.1)
    completion_start = start + timedelta(seconds=0.5)
    end = start + timedelta(seconds=1.0)
    kwargs = {
        "model": "gpt-4o-mini",
        "call_type": call_type,
        "litellm_params": ({"custom_llm_provider": provider} if provider is not None else {}),
        "optional_params": {"stream": stream},
        "response_cost": RESPONSE_COST,
        "api_call_start_time": api_call_start,
        "completion_start_time": completion_start,
        "end_time": end,
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": "hash-abc123",
                "user_api_key_team_id": "team-1",
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


def _drive_success(reader, callback_settings_attributes=None, **call_overrides):
    """Construct a metrics-on logger, optionally populate callback_settings AFTER
    construction (mirroring the proxy ordering), run the real success hook."""
    logger = _logger(reader, enable_metrics=True)
    previous = litellm.callback_settings
    if callback_settings_attributes is not None:
        litellm.callback_settings = {
            "otel": {"attributes": callback_settings_attributes}
        }
    try:
        kwargs, response_obj, start, end = _build_call(**call_overrides)
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


def test_response_read_does_not_replay_the_generation_usage():
    """A responses-management read returns the ORIGINAL generation's usage on the
    object it fetches. Recording it would add those tokens again on every poll, so
    the two usage-derived instruments are skipped while the duration ones, which
    describe the read itself, still fire."""
    metrics = _drive_success(InMemoryMetricReader(), call_type="aget_responses")

    assert TOKEN_USAGE not in metrics
    assert TIME_PER_OUTPUT_TOKEN not in metrics
    assert OPERATION_DURATION in metrics
    assert RESPONSE_DURATION in metrics


def test_background_response_read_still_records_usage():
    """A background=true create returns no usage, so its completed read is the only
    place the generation's tokens are ever seen. Skipping it would lose them
    entirely rather than deduplicate them."""
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    kwargs, response_obj, start, end = _build_call(call_type="aget_responses")
    response_obj["background"] = True
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    metrics = _metrics_by_name(reader)
    by_type = {dp.attributes[TOKEN_TYPE]: dp for dp in metrics[TOKEN_USAGE]}
    assert by_type["input"].sum == PROMPT_TOKENS
    assert by_type["output"].sum == COMPLETION_TOKENS
    assert TIME_PER_OUTPUT_TOKEN in metrics


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
    """exclude_list set AFTER construction (the proxy path) removes every listed
    key from more than one metric while the low-cardinality model attribute
    survives."""
    metrics = _drive_success(
        InMemoryMetricReader(),
        callback_settings_attributes={"exclude_list": list(FILTERABLE_KEYS)},
    )
    excluded = set(FILTERABLE_KEYS)

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


def test_no_filter_still_keeps_the_filterable_keys():
    """Without an attributes config every key the filter tests remove is present,
    so those tests prove a real removal rather than a key that was never there."""
    metrics = _drive_success(InMemoryMetricReader())
    expected = set(FILTERABLE_KEYS)

    for name in (OPERATION_DURATION, TOKEN_USAGE):
        for dp in metrics[name]:
            assert expected.issubset(set(dp.attributes.keys()))


def test_a_metric_ineligible_filter_name_is_reported_not_silently_dropped(caplog):
    """Naming a metric-ineligible attribute in a filter has to say so.

    The shared validator accepts every span attribute name, so an operator can put
    one in an ``include_list``, get nothing for it, and have no way to tell that from
    a value that happened to be absent. The ceiling is deliberate, but silent is what
    makes it a support ticket.
    """
    with caplog.at_level("WARNING"):
        _drive_success(
            InMemoryMetricReader(),
            callback_settings_attributes={
                "include_list": [MODEL_KEY, "metadata.requester_ip_address"]
            },
        )

    reported = [
        r.getMessage().split(" cannot be a metric attribute")[0].removeprefix("OTel metrics: ")
        for r in caplog.records
        if r.levelname == "WARNING" and "cannot be a metric attribute" in r.getMessage()
    ]
    assert reported == ["metadata.requester_ip_address"], reported


def test_two_calls_differing_only_per_request_share_one_series():
    """The whole point of the ceiling: metric cardinality must not grow with traffic.

    Every field here moves on every real request -- the response cost, the call id,
    the cache key, the provider's remaining-rate-limit headers -- and each one used
    to reach the datapoint inside a single ``hidden_params`` label. A unique label
    value is a new time series, so each of the six instruments minted one series per
    request, which is both a Grafana Cloud bill proportional to traffic and a
    histogram that cannot be aggregated. Identical attribute sets is what "one
    series" means to the SDK.
    """
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)

    for index, cost in enumerate((RESPONSE_COST, RESPONSE_COST * 3)):
        kwargs, response_obj, start, end = _build_call()
        kwargs["response_cost"] = cost
        kwargs["standard_logging_object"]["hidden_params"] = {
            "model_id": "m-1",
            # A documented per-call parameter, so it varies here on purpose: the same
            # deployment reached under a caller-chosen base must not split the series.
            "api_base": f"https://proxy-{index}.example.com/v1",
            "litellm_call_id": f"call-{index}",
            "cache_key": f"cache-{index}",
            "response_cost": cost,
            "litellm_overhead_time_ms": 1.5 + index,
            "usage_object": {"prompt_tokens": index, "completion_tokens": index},
            "additional_headers": {"x_ratelimit_remaining_requests": 100 - index},
        }
        asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    for name in ALL_METRICS:
        attribute_sets = {
            tuple(sorted((k, v) for k, v in dp.attributes.items() if k != TOKEN_TYPE))
            for dp in _metrics_by_name(reader)[name]
        }
        assert len(attribute_sets) == 1, f"{name} split into {len(attribute_sets)} series across 2 requests"


def test_hidden_params_label_carries_only_bounded_deployment_fields():
    """``hidden_params`` survives the ceiling, but only as the deployment identity.

    ``model_id`` is the router's deployment id, bounded by the deployment list, and is
    what a per-deployment dashboard reads. Everything else in the object is
    per-request or caller-chosen and belongs on the span, which already carries it.
    ``api_base`` is excluded despite naming the same deployment: it is a documented
    per-call parameter, so a caller varying it would restore the per-request
    cardinality this cap exists to remove.
    """
    kwargs, response_obj, start, end = _build_call()
    kwargs["standard_logging_object"]["hidden_params"] = {
        "model_id": "m-1",
        "api_base": "https://api.openai.com/v1",
        "litellm_call_id": "abc",
        "cache_key": "ck-1",
        "response_cost": RESPONSE_COST,
    }
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))

    label = _metrics_by_name(reader)[OPERATION_DURATION][0].attributes["hidden_params"]
    assert json.loads(label) == {"model_id": "m-1"}


def test_success_attributes_are_capped_at_the_ceiling():
    """The success path carries exactly the ceiling, no client-supplied attributes.

    The fixture deliberately sets every excluded key, so this asserts a real removal
    rather than keys that were never present.
    """
    kwargs, response_obj, start, end = _build_call()
    metadata = kwargs["standard_logging_object"]["metadata"]
    metadata.update(
        {
            "spend_logs_metadata": {"cost_center": "abc"},
            "user_api_key_end_user_id": "end-user-1",
            "user_api_key_user_email": "someone@example.com",
        }
    )
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    asyncio.run(logger.async_log_success_event(kwargs, response_obj, start, end))
    metrics = _metrics_by_name(reader)

    for name in ALL_METRICS:
        for dp in metrics[name]:
            leaked = set(dp.attributes) - set(BOUNDED_KEYS) - {TOKEN_TYPE}
            assert not leaked, f"{name} leaked {leaked}"
def test_provider_is_labelled_with_semconv_provider_name():
    """Every recorded point carries gen_ai.provider.name holding the semconv
    provider value (bedrock -> aws.bedrock), the key the GenAI convention and the
    dashboards built on it query. The deprecated gen_ai.system spelling alone is
    unreadable to them."""
    metrics = _drive_success(InMemoryMetricReader(), provider="bedrock")

    for name in ALL_METRICS:
        points = metrics[name]
        assert points, f"{name} was not recorded"
        for dp in points:
            assert dp.attributes[PROVIDER_NAME_KEY] == "aws.bedrock"


def test_deprecated_gen_ai_system_is_dual_emitted_verbatim():
    """gen_ai.system keeps its raw litellm provider value alongside the new key
    for one release, so a dashboard already filtering on it keeps matching. Its
    value must not be swapped for the mapped one, which would break exactly the
    queries the dual emission exists to protect."""
    metrics = _drive_success(InMemoryMetricReader(), provider="bedrock")

    for dp in metrics[OPERATION_DURATION]:
        assert dp.attributes[SYSTEM_KEY] == "bedrock"
        assert dp.attributes[PROVIDER_NAME_KEY] == "aws.bedrock"


def test_no_provider_attribute_when_provider_is_absent():
    """A call litellm could not attribute to a provider carries no provider label
    at all. A placeholder value ("Unknown") would mint a permanent series that
    aggregates every unattributable request and that no operator can act on."""
    metrics = _drive_success(InMemoryMetricReader(), provider=None)

    for name in ALL_METRICS:
        points = metrics[name]
        assert points, f"{name} was not recorded"
        for dp in points:
            keys = set(dp.attributes.keys())
            assert PROVIDER_NAME_KEY not in keys
            assert SYSTEM_KEY not in keys
            assert "Unknown" not in set(dp.attributes.values())


def test_vector_store_search_is_not_labelled_as_chat():
    """A vector-store search records under gen_ai.operation.name=retrieval, so its
    latency and cost stay out of the chat series."""
    metrics = _drive_success(InMemoryMetricReader(), call_type="avector_store_search")

    for name in (OPERATION_DURATION, TOKEN_COST):
        for dp in metrics[name]:
            assert dp.attributes[OPERATION_KEY] == "retrieval"


@pytest.mark.parametrize(
    "call_type,expected",
    [
        ("avector_store_create", "litellm.vector_store_management"),
        ("avector_store_delete", "litellm.vector_store_management"),
        ("avector_store_file_create", "litellm.vector_store_file_management"),
        ("avector_store_file_list", "litellm.vector_store_file_management"),
    ],
)
def test_vector_store_management_is_not_labelled_as_chat(call_type, expected):
    """Store and file management reach the recorder through the same success hook as a
    completion, so leaving them unmapped kept billing- and latency-relevant admin calls
    inside the chat series."""
    metrics = _drive_success(InMemoryMetricReader(), call_type=call_type)

    for dp in metrics[OPERATION_DURATION]:
        assert dp.attributes[OPERATION_KEY] == expected


@pytest.mark.parametrize("call_type", ["asend_message", "asend_message_streaming"])
def test_agent_message_is_not_labelled_as_chat(call_type):
    """An A2A agent send records under gen_ai.operation.name=invoke_agent, streamed or
    not. The streaming iterator dispatches the same success handlers under its own
    ``asend_message_streaming`` call type, so an unmapped streaming spelling puts every
    streamed agent turn's latency and cost back into the chat series."""
    metrics = _drive_success(InMemoryMetricReader(), call_type=call_type)

    for name in (OPERATION_DURATION, TOKEN_COST):
        for dp in metrics[name]:
            assert dp.attributes[OPERATION_KEY] == "invoke_agent"


def test_provider_name_is_filterable():
    """gen_ai.provider.name is a member of the metric-attribute allowlist, so an
    operator can include or exclude it; an unlisted name raises instead."""
    metrics = _drive_success(
        InMemoryMetricReader(),
        callback_settings_attributes={"include_list": [PROVIDER_NAME_KEY]},
    )

    for dp in metrics[OPERATION_DURATION]:
        assert set(dp.attributes.keys()) == {PROVIDER_NAME_KEY}


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
    with pytest.raises(ValueError, match='otel\\.attributes: gen_ai\\.token\\.type is a structural') as exc_info:
        recorder.record(kwargs, response_obj, start, end)
    # The dedicated discriminator guard, not the generic unknown-name path: assert
    # the specific reason so dropping that guard (and falling through to "unknown
    # attribute name") is caught.
    assert "discriminator" in str(exc_info.value)


# --- failure path ------------------------------------------------------------ #

ERROR_TYPE = "error.type"
ERROR_CLASS = "RateLimitError"
FAILURE_DURATION_S = 1.0

# Attributes a failure datapoint must never carry. Each is either supplied by the
# caller (so a caller could mint a fresh series per request, and a failure costs
# them no provider spend) or varies per request, or is PII duplicating an id that
# is already on the series.
UNBOUNDED_KEYS = (
    "metadata.requester_metadata",
    "metadata.requester_ip_address",
    "metadata.spend_logs_metadata",
    "metadata.user_api_key_end_user_id",
    "metadata.user_api_key_user_email",
)

# The exact set a datapoint may carry on either path: the operation, the
# operator-provisioned identity, and the deployment that served it.
BOUNDED_KEYS = (
    "hidden_params",
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.system",
    "gen_ai.request.model",
    "gen_ai.framework",
    "metadata.user_api_key_hash",
    "metadata.user_api_key_alias",
    "metadata.user_api_key_team_id",
    "metadata.user_api_key_team_alias",
    "metadata.user_api_key_org_id",
    "metadata.user_api_key_user_id",
)


def _build_failure(
    *,
    error_information=None,
    exception=None,
    no_upstream_call=False,
):
    """A captured failure-call ``(kwargs, start, end)``.

    Mirrors what litellm actually hands ``async_log_failure_event``: no
    ``response_obj`` at all, but the streaming timings and the recovered
    ``response_cost`` a mid-stream failure still carries -- so routing the failure
    path through the full success recorder would show up here as extra series
    rather than passing unnoticed. The metadata carries both the bounded identity
    keys and every caller-supplied / per-request key, so the allowlist test below
    proves a real removal rather than a key that was never there.
    """
    start = datetime(2026, 6, 12, 12, 0, 0)
    api_call_start = start + timedelta(seconds=0.1)
    completion_start = start + timedelta(seconds=0.5)
    end = start + timedelta(seconds=FAILURE_DURATION_S)
    standard_logging_object = {
        "status": "failure",
        "metadata": {
            "user_api_key_hash": "hash-abc123",
            "user_api_key_alias": "alias-abc",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "team-alpha",
            "user_api_key_org_id": "org-1",
            "user_api_key_user_id": "user-1",
            "user_api_key_user_email": "user@example.com",
            "user_api_key_end_user_id": "end-user-42",
            "requester_ip_address": "10.0.0.7",
            "requester_metadata": {"trace": "caller-supplied-unique-value"},
            "spend_logs_metadata": {"ticket": "caller-supplied-unique-value"},
        },
        "hidden_params": {
            "litellm_call_id": "abc",
            "model_id": "m-1",
            "api_base": "https://api.openai.com/v1",
        },
    }
    if error_information is not None:
        standard_logging_object["error_information"] = error_information
    kwargs = {
        "model": "gpt-4o-mini",
        "call_type": "completion",
        "litellm_params": {"custom_llm_provider": "openai"},
        "optional_params": {"stream": True},
        "response_cost": RESPONSE_COST,
        "api_call_start_time": api_call_start,
        "completion_start_time": completion_start,
        "end_time": end,
        "standard_logging_object": standard_logging_object,
    }
    if exception is not None:
        kwargs["exception"] = exception
    if no_upstream_call:
        kwargs[LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL] = True
    return kwargs, start, end


def _drive_failure(reader, callback_settings_attributes=None, **failure_kwargs):
    logger = _logger(reader, enable_metrics=True)
    previous = litellm.callback_settings
    if callback_settings_attributes is not None:
        litellm.callback_settings = {"otel": {"attributes": callback_settings_attributes}}
    try:
        kwargs, start, end = _build_failure(**failure_kwargs)
        asyncio.run(logger.async_log_failure_event(kwargs, None, start, end))
    finally:
        litellm.callback_settings = previous
    return _metrics_by_name(reader)


def test_failure_records_only_the_duration_histogram():
    """A failed call contributes to gen_ai.client.operation.duration -- before this
    existed a failure recorded nothing at all, so the histogram measured only the
    traffic that survived. It contributes to nothing else: the other five
    instruments describe a completed generation, and the call carries a streaming
    timing pair and a recovered response_cost that would light four of them up if
    the failure were routed through the success recorder."""
    metrics = _drive_failure(
        InMemoryMetricReader(),
        error_information={"error_class": ERROR_CLASS, "error_code": "429"},
    )

    assert set(metrics.keys()) == {OPERATION_DURATION}
    points = metrics[OPERATION_DURATION]
    assert len(points) == 1
    assert points[0].count == 1
    assert points[0].sum == pytest.approx(FAILURE_DURATION_S)
    assert points[0].attributes[ERROR_TYPE] == ERROR_CLASS


def test_success_and_failure_are_separable_and_success_attributes_unchanged():
    """The pooled histogram stays queryable per outcome, and the existing
    dashboards keep working.

    A success and a failure through one reader must land on two distinct series --
    one with error.type, one without -- so a failure-rate panel is expressible and
    an operator can still get success-only latency by filtering error.type="". The
    success datapoint's attribute map must be byte-for-byte the map a success-only
    run produces, which is what stops the new attribute from leaking onto the
    series every current query reads."""
    baseline_reader = InMemoryMetricReader()
    baseline = _drive_success(baseline_reader)
    baseline_points = baseline[OPERATION_DURATION]
    assert len(baseline_points) == 1
    baseline_attributes = dict(baseline_points[0].attributes)

    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    ok_kwargs, response_obj, ok_start, ok_end = _build_call()
    asyncio.run(logger.async_log_success_event(ok_kwargs, response_obj, ok_start, ok_end))
    bad_kwargs, bad_start, bad_end = _build_failure(error_information={"error_class": ERROR_CLASS})
    asyncio.run(logger.async_log_failure_event(bad_kwargs, None, bad_start, bad_end))

    points = metrics = _metrics_by_name(reader)[OPERATION_DURATION]
    assert len(points) == 2, f"success and failure collapsed into {len(points)} series: {metrics}"
    succeeded = [dp for dp in points if ERROR_TYPE not in dp.attributes]
    failed = [dp for dp in points if dp.attributes.get(ERROR_TYPE) == ERROR_CLASS]
    assert len(succeeded) == 1 and len(failed) == 1
    assert dict(succeeded[0].attributes) == baseline_attributes


def test_failure_attributes_are_a_bounded_allowlist():
    """A failure datapoint carries exactly the bounded allowlist plus error.type.

    A failed request needs no provider spend, so nothing rate-limits a caller who
    puts a unique value into an attribute they control and mints one histogram
    series per request. The same payload is driven through the success path first,
    which does carry those keys, so this asserts a real removal on the failure path
    rather than keys that were never present. The exact-set assertion is the guard
    against the natural refactor of "just reuse _common_attributes"."""
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    kwargs, start, end = _build_failure(error_information={"error_class": ERROR_CLASS})
    usage = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    asyncio.run(logger.async_log_success_event(kwargs, usage, start, end))
    asyncio.run(logger.async_log_failure_event(kwargs, None, start, end))

    points = _metrics_by_name(reader)[OPERATION_DURATION]
    succeeded = next(dp for dp in points if ERROR_TYPE not in dp.attributes)
    failed = next(dp for dp in points if ERROR_TYPE in dp.attributes)

    supplied = set(kwargs["standard_logging_object"]["metadata"])
    missing = {key for key in UNBOUNDED_KEYS if key.removeprefix("metadata.") not in supplied}
    assert not missing, f"fixture never carried {missing}, so the exclusion below proves nothing"
    leaked = set(UNBOUNDED_KEYS) & set(failed.attributes)
    assert not leaked, f"failure datapoint leaked unbounded attributes: {leaked}"
    assert set(failed.attributes) == set(BOUNDED_KEYS) | {ERROR_TYPE}
    assert json.loads(failed.attributes["hidden_params"]) == {"model_id": "m-1"}


def test_operator_filter_can_still_narrow_the_failure_allowlist():
    """The allowlist is a ceiling, not a floor: an exclude_list an operator sets
    still removes a listed key from the failure series."""
    metrics = _drive_failure(
        InMemoryMetricReader(),
        callback_settings_attributes={"exclude_list": ["metadata.user_api_key_hash"]},
        error_information={"error_class": ERROR_CLASS},
    )
    attributes = metrics[OPERATION_DURATION][0].attributes
    assert "metadata.user_api_key_hash" not in attributes
    assert attributes[ERROR_TYPE] == ERROR_CLASS
    assert attributes[MODEL_KEY] == "gpt-4o-mini"


@pytest.mark.parametrize(
    "failure_kwargs, expected",
    [
        ({"error_information": {"error_class": ERROR_CLASS, "error_code": "429"}}, ERROR_CLASS),
        ({"error_information": {"error_code": "429"}}, "429"),
        ({"exception": ValueError("boom")}, "ValueError"),
        ({}, "_OTHER"),
    ],
    ids=["error_class", "error_code_only", "exception_fallback", "unclassifiable"],
)
def test_error_type_is_bounded_and_falls_back(failure_kwargs, expected):
    """error.type is always a bounded value: the mapped exception's class name, the
    provider status code, the raw exception's class name, or the semconv _OTHER
    fallback. Never the exception message, which is unbounded."""
    metrics = _drive_failure(InMemoryMetricReader(), **failure_kwargs)
    assert metrics[OPERATION_DURATION][0].attributes[ERROR_TYPE] == expected


def test_include_list_cannot_strip_error_type():
    """error.type is a structural discriminator like gen_ai.token.type: an
    include_list that does not mention it must not merge the failure series back
    into the success series, so it is stamped after the filter runs."""
    metrics = _drive_failure(
        InMemoryMetricReader(),
        callback_settings_attributes={"include_list": [MODEL_KEY]},
        error_information={"error_class": ERROR_CLASS},
    )
    attributes = metrics[OPERATION_DURATION][0].attributes
    assert dict(attributes) == {MODEL_KEY: "gpt-4o-mini", ERROR_TYPE: ERROR_CLASS}


def test_proxy_gate_rejection_records_no_duration():
    """A synthetic proxy-gate failure log (auth / rate-limit rejection) never made
    an upstream call, so its wall time is not a GenAI operation's duration; it is
    skipped for the same reason it gets no span. Recording it would pull the
    histogram toward the proxy's own latency.

    Both failures go through one reader so the assertion is that exactly the
    upstream one landed, rather than the vacuous "nothing was recorded" a
    failure path that records nothing at all would also satisfy."""
    reader = InMemoryMetricReader()
    logger = _logger(reader, enable_metrics=True)
    gate_kwargs, gate_start, gate_end = _build_failure(
        error_information={"error_class": "AuthenticationError"},
        no_upstream_call=True,
    )
    asyncio.run(logger.async_log_failure_event(gate_kwargs, None, gate_start, gate_end))
    upstream_kwargs, upstream_start, upstream_end = _build_failure(error_information={"error_class": ERROR_CLASS})
    asyncio.run(logger.async_log_failure_event(upstream_kwargs, None, upstream_start, upstream_end))

    points = _metrics_by_name(reader)[OPERATION_DURATION]
    assert [dp.attributes[ERROR_TYPE] for dp in points] == [ERROR_CLASS]
    assert points[0].count == 1
