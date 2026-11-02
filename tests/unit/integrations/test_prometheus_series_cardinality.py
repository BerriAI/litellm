import logging
import re
from pathlib import Path
from threading import Thread
from typing import Final

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, Counter, generate_latest

import litellm
from litellm.constants import PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX
from litellm.integrations.prometheus import PrometheusLogger, _LabeledMetric, prometheus_label_factory
from litellm.integrations.prometheus_helpers import bounded_prometheus_series_tracker
from litellm.integrations.prometheus_helpers.bounded_prometheus_series_tracker import (
    BoundedPrometheusSeriesTracker,
    PrometheusSeriesLimits,
)
from litellm.integrations.prometheus_helpers.shared_prometheus_series_admissions import (
    SharedPrometheusSeriesAdmissions,
)
from litellm.proxy.prometheus_cleanup import wipe_directory
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
    collectors_before: Final = tuple(REGISTRY._collector_to_names)
    _unregister_everything()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    for setting in SERIES_SETTINGS:
        monkeypatch.setattr(litellm, setting, getattr(litellm, setting))
    yield
    _unregister_everything()
    for collector in collectors_before:
        REGISTRY.register(collector)


@pytest.fixture
def clock(monkeypatch):
    now: Final = [1_000.0]
    monkeypatch.setattr(bounded_prometheus_series_tracker.time, "monotonic", lambda: now[0])
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


def test_workers_sharing_a_multiprocess_dir_admit_the_same_label_sets(tmp_path: Path):
    first_worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    second_worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    assert first_worker.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
    assert second_worker.admit_series("litellm_requests_metric", ("user-b",), max_series=2)

    replacement_worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    for worker in (first_worker, second_worker, replacement_worker):
        assert worker.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
        assert worker.admit_series("litellm_requests_metric", ("user-b",), max_series=2)
        assert not worker.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert first_worker.admit_series("litellm_spend_metric", ("user-c",), max_series=2)


def test_workers_agree_when_racing_appends_overfill_the_admissions_file(tmp_path: Path):
    racing_workers: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    for user in ("user-a", "user-b", "user-c"):
        assert racing_workers.admit_series("litellm_requests_metric", (user,), max_series=3)

    worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    assert not worker.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert worker.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
    assert worker.admit_series("litellm_requests_metric", ("user-b",), max_series=2)


def test_a_line_another_worker_is_still_writing_is_read_once_it_is_complete(tmp_path: Path):
    admissions_file: Final = tmp_path / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric"
    assert SharedPrometheusSeriesAdmissions(directory=str(tmp_path)).admit_series(
        "litellm_requests_metric", ("user-a",), max_series=2
    )
    reader: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    with admissions_file.open("ab") as write_in_progress:
        write_in_progress.write(b'["user')
        write_in_progress.flush()
        assert reader.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
        write_in_progress.write(b'-b"]\n')

    assert not reader.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert reader.admit_series("litellm_requests_metric", ("user-b",), max_series=2)


def test_a_record_cut_short_by_a_full_disk_admits_nothing_and_hides_no_other_record(tmp_path: Path):
    admissions_file: Final = tmp_path / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric"
    admissions_file.write_bytes(b'\n["user-a')
    writer: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    reader: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    assert writer.admit_series("litellm_requests_metric", ("user-b",), max_series=2)
    assert reader.admit_series("litellm_requests_metric", ("user-b",), max_series=2)
    assert reader.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert writer.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert not writer.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
    assert not reader.admit_series("litellm_requests_metric", ("user-a",), max_series=2)


def test_wiping_the_multiprocess_dir_frees_every_admitted_slot(tmp_path: Path):
    before_restart: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    assert before_restart.admit_series("litellm_requests_metric", ("user-a",), max_series=1)

    wipe_directory(str(tmp_path))

    after_restart: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    assert after_restart.admit_series("litellm_requests_metric", ("user-b",), max_series=1)
    assert not after_restart.admit_series("litellm_requests_metric", ("user-a",), max_series=1)


def test_eviction_racing_a_new_series_cannot_leave_it_untracked():
    registry: Final = CollectorRegistry()
    counter: Final = Counter("requests", "requests", labelnames=("user",), registry=registry)

    class _EvictedWhileBeingCreated:
        def labels(self, *labelvalues: str):
            if labelvalues == ("evicted-user",):
                eviction.start()
                eviction.join(timeout=0.05)
            return counter.labels(*labelvalues)

        def remove(self, *labelvalues: str) -> None:
            counter.remove(*labelvalues)

    labeled: Final = _LabeledMetric(
        metric=_EvictedWhileBeingCreated(),
        metric_name="requests",
        original_labelnames=("user",),
        excluded_labels=frozenset(),
        tracker=BoundedPrometheusSeriesTracker(),
        limits=PrometheusSeriesLimits(max_series=1, ttl_seconds=None, cleanup_interval_seconds=None),
        shares_overflow_series=True,
    )
    eviction: Final = Thread(target=labeled.remove, args=("evicted-user",))

    labeled.labels("evicted-user").inc()
    eviction.join()
    labeled.labels("next-user").inc()

    assert _label_values(_scraped_series("requests_total", registry), "user") == {"next-user"}


@pytest.mark.parametrize(
    "metric_name", ["litellm_deployment_successful_fallbacks", "litellm_deployment_failed_fallbacks"]
)
def test_cap_applies_to_the_fallback_counters(metric_name: str):
    litellm.prometheus_metrics_max_series_per_metric = 2
    litellm.prometheus_metrics_ttl_seconds = None
    logger: Final = PrometheusLogger()

    for index in range(4):
        PrometheusLogger._inc_labeled_counter(
            logger,
            getattr(logger, metric_name),
            metric_name,
            UserAPIKeyLabelValues(fallback_model=f"model-{index}"),
        )

    series: Final = _scraped_series(f"{metric_name}_total")
    assert _label_values(series, "fallback_model") == {"model-0", "model-1", "other"}
    assert _sample_value(series, "fallback_model", "other") == 2


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
        ("prometheus_metrics_max_series_per_metric", "five"),
        ("prometheus_metrics_max_series_per_metric", True),
        ("prometheus_metrics_max_series_per_metric", 2.5),
        ("prometheus_metrics_ttl_seconds", ""),
    ],
)
def test_a_series_limit_that_is_not_a_positive_number_is_ignored_with_a_warning_and_metrics_keep_flowing(
    setting: str, value: object, clock, caplog
):
    litellm.prometheus_metrics_cleanup_interval_seconds = 0.0
    setattr(litellm, setting, value)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    for index in range(3):
        _count_request(logger, f"agent-{index}")
        clock[0] += 100.0

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-0", "agent-1", "agent-2"}
    assert setting in caplog.text


@pytest.mark.parametrize("value", ["sixty", -1, True, ""])
def test_a_cleanup_interval_that_is_not_a_number_of_at_least_zero_falls_back_to_the_default_with_a_warning(
    value: object, monkeypatch, clock, caplog
):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 3)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_metrics_cleanup_interval_seconds", value)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    _count_request(logger, "agent-0")
    clock[0] += 30.0
    _count_request(logger, "agent-1")
    within_the_interval: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    clock[0] += 31.0
    _count_request(logger, "agent-2")

    assert "prometheus_metrics_cleanup_interval_seconds" in caplog.text
    assert _label_values(within_the_interval, "user_agent") == {"agent-0", "agent-1"}
    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-2"}


def test_a_cleanup_interval_written_as_a_numeric_string_is_honored(monkeypatch, clock, caplog):
    monkeypatch.setattr(litellm, "prometheus_metrics_max_series_per_metric", 3)
    monkeypatch.setattr(litellm, "prometheus_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_metrics_cleanup_interval_seconds", "0")

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    _count_request(logger, "agent-0")
    clock[0] += 30.0
    _count_request(logger, "agent-1")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-1"}
    assert "prometheus_metrics_cleanup_interval_seconds" not in caplog.text


def test_a_series_cap_written_as_a_numeric_string_is_honored(caplog):
    litellm.prometheus_metrics_max_series_per_metric = "2"

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        logger: Final = PrometheusLogger()
    for index in range(3):
        _count_request(logger, f"agent-{index}")

    series: Final = _scraped_series("litellm_proxy_total_requests_metric_total")
    assert _label_values(series, "user_agent") == {"agent-0", "agent-1", "other"}
    assert "prometheus_metrics_max_series_per_metric" not in caplog.text
