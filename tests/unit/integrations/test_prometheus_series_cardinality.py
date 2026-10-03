import re
from time import monotonic
from typing import Final

import pytest
from prometheus_client import REGISTRY, generate_latest

import litellm
from litellm.integrations.prometheus import PrometheusLogger, prometheus_label_factory
from litellm.integrations.prometheus_helpers import bounded_prometheus_series_tracker
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues

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
    _unregister_everything()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    for setting in SERIES_SETTINGS:
        monkeypatch.setattr(litellm, setting, getattr(litellm, setting))
    yield
    _unregister_everything()


@pytest.fixture
def clock(monkeypatch):
    now: Final = [monotonic()]
    monkeypatch.setattr(bounded_prometheus_series_tracker.time, "monotonic", lambda: now[0])
    return now


def _scraped_series(sample_name: str) -> frozenset[str]:
    exposition: Final = generate_latest(REGISTRY).decode()
    return frozenset(line for line in exposition.splitlines() if line.startswith(f"{sample_name}{{"))


def _label_values(series: frozenset[str], label: str) -> frozenset[str]:
    pattern: Final = re.compile(rf'[{{,]{label}="([^"]*)"')
    return frozenset(match.group(1) for match in map(pattern.search, series) if match is not None)


def _sample_value(series: frozenset[str], label: str, value: str) -> float:
    (line,) = (line for line in series if f'{label}="{value}"' in line)
    return float(line.rsplit(" ", 1)[1])


def _count_request(logger: PrometheusLogger, user_agent: str) -> None:
    PrometheusLogger._inc_labeled_counter(
        logger,
        logger.litellm_proxy_total_requests_metric,
        "litellm_proxy_total_requests_metric",
        UserAPIKeyLabelValues(user_agent=user_agent),
    )


def _observe_latency(logger: PrometheusLogger, user: str) -> None:
    labels: Final = prometheus_label_factory(
        supported_enum_labels=logger.get_labels_for_metric("litellm_request_total_latency_metric"),
        enum_values=UserAPIKeyLabelValues(user=user),
    )
    logger.litellm_request_total_latency_metric.labels(**labels).observe(0.5)


def test_label_sets_past_the_cap_are_counted_on_one_other_series():
    litellm.prometheus_metrics_max_series_per_metric = 2
    litellm.prometheus_metrics_ttl_seconds = None
    logger: Final = PrometheusLogger()

    for identity in ("one", "two", "three", "one", "four"):
        _count_request(logger, f"codex/{identity}")
        _observe_latency(logger, f"user-{identity}")

    counter_series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(counter_series, "user_agent") == {"codex/one", "codex/two", "other"}
    assert _sample_value(counter_series, "user_agent", "codex/one") == 2
    assert _sample_value(counter_series, "user_agent", "codex/two") == 1
    assert _sample_value(counter_series, "user_agent", "other") == 2
    histogram_series: Final = _scraped_series("litellm_request_total_latency_metric_count")
    assert _label_values(histogram_series, "user") == {"user-one", "user-two", "other"}
    assert _sample_value(histogram_series, "user", "other") == 2


def test_gauge_label_sets_past_the_cap_are_not_emitted():
    litellm.prometheus_metrics_max_series_per_metric = 2
    litellm.prometheus_metrics_ttl_seconds = None
    logger: Final = PrometheusLogger()

    for provider in ("openai", "anthropic", "bedrock", "openai"):
        logger.track_provider_remaining_budget(provider=provider, spend=1.0, budget_limit=10.0)

    series: Final = _scraped_series("litellm_provider_remaining_budget_metric")
    assert _label_values(series, "api_provider") == {"openai", "anthropic"}


def test_series_idle_past_the_ttl_are_removed_and_free_their_slot(clock):
    litellm.prometheus_metrics_max_series_per_metric = 2
    litellm.prometheus_metrics_ttl_seconds = 10.0
    litellm.prometheus_metrics_cleanup_interval_seconds = 0.0
    logger: Final = PrometheusLogger()

    _count_request(logger, "idle-agent")
    clock[0] += 9.0
    _count_request(logger, "still-fresh-agent")
    clock[0] += 2.0
    _count_request(logger, "new-agent")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"still-fresh-agent", "new-agent"}


def test_cap_holds_and_ttl_is_ignored_in_multiprocess_mode(monkeypatch, tmp_path, clock):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    litellm.prometheus_metrics_max_series_per_metric = 2
    litellm.prometheus_metrics_ttl_seconds = 10.0
    litellm.prometheus_metrics_cleanup_interval_seconds = 0.0
    logger: Final = PrometheusLogger()

    _count_request(logger, "first-agent")
    _count_request(logger, "second-agent")
    clock[0] += 11.0
    _count_request(logger, "third-agent")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"first-agent", "second-agent", "other"}


def test_end_user_eviction_keeps_the_series_and_its_slot_in_multiprocess_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    litellm.enable_end_user_cost_tracking_prometheus_only = True
    litellm.prometheus_metrics_config = [
        {"group": "end-user-spend", "metrics": ["litellm_spend_metric"], "include_labels": ["end_user"]}
    ]
    litellm.prometheus_end_user_metrics_max_series_per_metric = 2
    litellm.prometheus_end_user_metrics_ttl_seconds = None
    litellm.prometheus_metrics_max_series_per_metric = 3
    litellm.prometheus_metrics_ttl_seconds = None
    logger: Final = PrometheusLogger()

    for index in range(5):
        PrometheusLogger._inc_labeled_counter(
            logger,
            logger.litellm_spend_metric,
            "litellm_spend_metric",
            UserAPIKeyLabelValues(end_user=f"end-user-{index}"),
            amount=0.01,
        )

    series: Final = _scraped_series("litellm_spend_metric_total")
    assert _label_values(series, "end_user") == {"end-user-0", "end-user-1", "end-user-2", "other"}


def test_cap_applies_under_a_globally_excluded_label():
    litellm.prometheus_exclude_labels = ["hook_type"]
    litellm.prometheus_metrics_max_series_per_metric = 2
    litellm.prometheus_metrics_ttl_seconds = None
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


def test_end_user_eviction_frees_a_slot_under_the_cap():
    litellm.enable_end_user_cost_tracking_prometheus_only = True
    litellm.prometheus_metrics_config = [
        {"group": "end-user-spend", "metrics": ["litellm_spend_metric"], "include_labels": ["end_user"]}
    ]
    litellm.prometheus_end_user_metrics_max_series_per_metric = 2
    litellm.prometheus_end_user_metrics_ttl_seconds = None
    litellm.prometheus_metrics_max_series_per_metric = 3
    litellm.prometheus_metrics_ttl_seconds = None
    logger: Final = PrometheusLogger()

    for index in range(5):
        PrometheusLogger._inc_labeled_counter(
            logger,
            logger.litellm_spend_metric,
            "litellm_spend_metric",
            UserAPIKeyLabelValues(end_user=f"end-user-{index}"),
            amount=0.01,
        )

    series: Final = _scraped_series("litellm_spend_metric_total")
    assert _label_values(series, "end_user") == {"end-user-3", "end-user-4"}


def test_series_stay_unbounded_unless_a_limit_is_configured():
    litellm.prometheus_metrics_max_series_per_metric = None
    litellm.prometheus_metrics_ttl_seconds = None
    logger: Final = PrometheusLogger()

    for index in range(5):
        _count_request(logger, f"agent-{index}")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {f"agent-{index}" for index in range(5)}


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("prometheus_metrics_max_series_per_metric", 0),
        ("prometheus_metrics_max_series_per_metric", -5),
        ("prometheus_metrics_ttl_seconds", 0.0),
        ("prometheus_metrics_ttl_seconds", -1.0),
    ],
)
def test_non_positive_series_limits_fail_logger_startup(setting: str, value: float):
    setattr(litellm, setting, value)

    with pytest.raises(ValueError, match=setting):
        PrometheusLogger()
