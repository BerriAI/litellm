import logging
import re
from datetime import datetime
from time import time
from typing import Final

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_helpers import native_store
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_success,
    standard_payload,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


SERIES_SETTINGS: Final = (
    "prometheus_metrics_max_series_per_metric",
    "prometheus_metrics_ttl_seconds",
    "prometheus_metrics_cleanup_interval_seconds",
    "prometheus_exclude_labels",
    "prometheus_metrics_config",
    "enable_end_user_cost_tracking_prometheus_only",
    "prometheus_end_user_metrics_max_series_per_metric",
    "prometheus_end_user_metrics_ttl_seconds",
)


def _unregister_everything() -> None:
    for collector in list(REGISTRY._collector_to_names):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def isolated_registry_and_settings(monkeypatch):
    collectors_before: Final = tuple(REGISTRY._collector_to_names)
    _unregister_everything()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    for setting in SERIES_SETTINGS:
        monkeypatch.setattr(litellm, setting, getattr(litellm, setting))
    yield
    _unregister_everything()
    for collector in collectors_before:
        REGISTRY.register(collector)


class _StoreClock:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.stores: list[object] = []

    def __getitem__(self, index: int) -> float:
        assert index == 0
        return self.now

    def __setitem__(self, index: int, value: float) -> None:
        assert index == 0
        self.now = value
        for store in self.stores:
            store.set_time(value, time())


@pytest.fixture
def clock(monkeypatch):
    now: Final = _StoreClock()
    real_store: Final = native_store._native.NativePrometheusStore

    def build(max_series=None, ttl_seconds=None, cleanup_interval_seconds=None):
        store = real_store(max_series, ttl_seconds, cleanup_interval_seconds, manual_clock=True)
        store.set_time(now.now, time())
        now.stores.append(store)
        return store

    monkeypatch.setattr(native_store._native, "NativePrometheusStore", build)
    return now


def _scraped_series(sample_name: str, registry: CollectorRegistry = REGISTRY) -> frozenset[str]:
    exposition: Final = generate_latest(registry).decode()
    return frozenset(line for line in exposition.splitlines() if line.startswith(f"{sample_name}{{"))


def _label_values(series: frozenset[str], label: str) -> frozenset[str]:
    pattern: Final = re.compile(rf'[{{,]{label}="([^"]*)"')
    return frozenset(match.group(1) for match in map(pattern.search, series) if match is not None)


def _sample_value(series: frozenset[str], label: str, value: str) -> float:
    (line,) = (line for line in series if f'{label}="{value}"' in line)
    return float(line.rsplit(" ", 1)[1])


async def _request(
    logger: PrometheusLogger, *, user_agent: str = "agent", user: str = "user", end_user: str | None = None
) -> None:
    now: Final = datetime.now()
    kwargs: Final = {
        "model": "gpt-4",
        "litellm_params": {"metadata": {"user_api_key_end_user_id": end_user}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": standard_payload(
            metadata={"user_agent": user_agent, "user_api_key_user_id": user},
            response_cost=0.01,
            end_user=end_user,
        ),
    }
    await native_success(logger, kwargs, None)


async def _count_request(logger: PrometheusLogger, user_agent: str) -> None:
    await _request(logger, user_agent=user_agent)


@pytest.mark.asyncio
async def test_label_sets_past_the_cap_are_counted_on_one_other_series(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", None)
    logger: Final = PrometheusLogger()

    for identity in ("one", "two", "three", "one", "four"):
        await _request(logger, user_agent=f"codex/{identity}", user=f"user-{identity}")

    counter_series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(counter_series, "user_agent") == {"codex/one", "codex/two", "other"}
    assert _sample_value(counter_series, "user_agent", "codex/one") == 2
    assert _sample_value(counter_series, "user_agent", "codex/two") == 1
    assert _sample_value(counter_series, "user_agent", "other") == 2
    histogram_series: Final = _scraped_series("litellm_request_total_latency_metric_count")
    assert _label_values(histogram_series, "user") == {"user-one", "user-two", "other"}
    assert _sample_value(histogram_series, "user", "other") == 2


def test_gauge_label_sets_past_the_cap_are_not_emitted(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", None)
    logger: Final = PrometheusLogger()

    for provider in ("openai", "anthropic", "bedrock", "openai"):
        logger.track_provider_remaining_budget(provider=provider, spend=1.0, budget_limit=10.0)

    series: Final = _scraped_series("litellm_provider_remaining_budget_metric")
    assert _label_values(series, "api_provider") == {"openai", "anthropic"}


@pytest.mark.asyncio
async def test_series_idle_past_the_ttl_are_removed_and_free_their_slot(clock, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_metrics_cleanup_interval_seconds", 0.0)
    logger: Final = PrometheusLogger()

    await _count_request(logger, "idle-agent")
    clock[0] += 9.0
    await _count_request(logger, "still-fresh-agent")
    clock[0] += 2.0
    await _count_request(logger, "new-agent")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"still-fresh-agent", "new-agent"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_name", ["litellm_deployment_successful_fallbacks", "litellm_deployment_failed_fallbacks"]
)
async def test_cap_applies_to_the_fallback_counters(metric_name: str, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", None)
    logger: Final = PrometheusLogger()
    log_event: Final = (
        logger.log_success_fallback_event
        if metric_name == "litellm_deployment_successful_fallbacks"
        else logger.log_failure_fallback_event
    )

    for index in range(4):
        await log_event(
            original_model_group="primary",
            kwargs={"model": f"model-{index}", "metadata": {}},
            original_exception=Exception("upstream unavailable"),
        )

    series: Final = _scraped_series(f"{metric_name}_total")
    assert _label_values(series, "fallback_model") == {"model-0", "model-1", "other"}
    assert _sample_value(series, "fallback_model", "other") == 2


def test_cap_applies_under_a_globally_excluded_label(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_exclude_labels", ["hook_type"])
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", None)
    logger: Final = PrometheusLogger()

    for index in range(4):
        logger._record_guardrail_metrics(
            guardrail_name=f"guardrail-{index}",
            latency_seconds=0.1,
            status="success",
            error_type=None,
            hook_type="pre_call",
        )

    series: Final = _scraped_series("litellm_guardrail_requests_total")
    assert _label_values(series, "guardrail_name") == {"guardrail-0", "guardrail-1", "other"}
    assert _sample_value(series, "guardrail_name", "other") == 2
    assert all("hook_type" not in line for line in series)


@pytest.mark.asyncio
async def test_end_user_eviction_frees_a_slot_under_the_cap(monkeypatch):
    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", True)
    monkeypatch.setattr(
        litellm,
        "prometheus_metrics_config",
        [{"group": "end-user-spend", "metrics": ["litellm_spend_metric"], "include_labels": ["end_user"]}],
    )
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_ttl_seconds", None)
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 3)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", None)
    logger: Final = PrometheusLogger()

    for index in range(5):
        await _request(logger, end_user=f"end-user-{index}")

    series: Final = _scraped_series("litellm_spend_metric_total")
    assert _label_values(series, "end_user") == {"end-user-3", "end-user-4"}


@pytest.mark.asyncio
async def test_series_stay_unbounded_unless_a_limit_is_configured(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", None)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", None)
    logger: Final = PrometheusLogger()

    for index in range(5):
        await _count_request(logger, f"agent-{index}")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {f"agent-{index}" for index in range(5)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("prometheus_metrics_max_series_per_metric", 0),
        ("prometheus_metrics_max_series_per_metric", -5),
        ("prometheus_metrics_ttl_seconds", 0.0),
        ("prometheus_metrics_ttl_seconds", -1.0),
        ("prometheus_metrics_max_series_per_metric", "five"),
        ("prometheus_metrics_max_series_per_metric", True),
        ("prometheus_metrics_max_series_per_metric", 2.5),
        ("prometheus_metrics_ttl_seconds", ""),
    ],
)
async def test_a_series_limit_that_is_not_a_positive_number_is_ignored_with_a_warning_and_metrics_keep_flowing(
    setting: str, value: object, clock, caplog, monkeypatch
):
    monkeypatch.setattr(litellm, "prometheus_metrics_cleanup_interval_seconds", 0.0)
    monkeypatch.setattr(litellm, setting, value)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    for index in range(3):
        await _count_request(logger, f"agent-{index}")
        clock[0] += 100.0

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-0", "agent-1", "agent-2"}
    assert setting in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["sixty", -1, True, ""])
async def test_a_cleanup_interval_that_is_not_a_number_of_at_least_zero_falls_back_to_the_default_with_a_warning(
    value: object, monkeypatch, clock, caplog
):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 3)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_metrics_cleanup_interval_seconds", value)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    await _count_request(logger, "agent-0")
    clock[0] += 30.0
    await _count_request(logger, "agent-1")
    within_the_interval: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    clock[0] += 31.0
    await _count_request(logger, "agent-2")

    assert "prometheus_metrics_cleanup_interval_seconds" in caplog.text
    assert _label_values(within_the_interval, "user_agent") == {"agent-0", "agent-1"}
    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-2"}


@pytest.mark.asyncio
async def test_a_cleanup_interval_written_as_a_numeric_string_is_honored(monkeypatch, clock, caplog):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 3)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_metrics_cleanup_interval_seconds", "0")

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    await _count_request(logger, "agent-0")
    clock[0] += 30.0
    await _count_request(logger, "agent-1")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-1"}
    assert "prometheus_metrics_cleanup_interval_seconds" not in caplog.text


@pytest.mark.asyncio
async def test_a_series_cap_written_as_a_numeric_string_is_honored(caplog, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", "2")

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    for index in range(3):
        await _count_request(logger, f"agent-{index}")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-0", "agent-1", "other"}
    assert "prometheus_metrics_max_series_per_metric" not in caplog.text
