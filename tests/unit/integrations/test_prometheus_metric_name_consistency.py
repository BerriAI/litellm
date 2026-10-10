"""
Unit tests for prometheus metric name consistency

This test ensures that the metric names used when creating Prometheus metrics
match the names defined in DEFINED_PROMETHEUS_METRICS, so that metric filtering
configuration works correctly.

Related issue: https://github.com/BerriAI/litellm/issues/18221
"""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final, get_args

import pytest
from prometheus_client import REGISTRY, Gauge
from prometheus_client.registry import Collector

import litellm
from litellm.caching.redis_cache import _breaker_metrics
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_services import PrometheusServicesLogger
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup_metrics import SpendLogCleanupMetrics
from litellm.proxy.middleware.admission_control_middleware import create_prometheus_admission_metrics
from litellm.proxy.middleware.in_flight_requests_middleware import InFlightRequestsMiddleware

_GRAFANA_DIR: Final = Path(__file__).parents[3] / "cookbook" / "litellm_proxy_server" / "grafana_dashboard"
_ALL_METRICS_DASHBOARD: Final = _GRAFANA_DIR / "dashboard_all_metrics" / "grafana_dashboard.json"
_LITELLM_DASHBOARDS: Final = (_ALL_METRICS_DASHBOARD, _GRAFANA_DIR / "dashboard_v2" / "grafana_dashboard.json")
_METRIC_TOKEN_RE: Final = re.compile(r"\blitellm_[a-z0-9_]+")
_BY_CLAUSE_RE: Final = re.compile(r"\bby\s*\([^)]*\)")
_EXPOSITION_SUFFIXES: Final = ("", "_total", "_bucket", "_sum", "_count", "_created")


def _registered_collectors() -> MappingProxyType[Collector, tuple[str, ...]]:
    return MappingProxyType({collector: tuple(names) for collector, names in REGISTRY._collector_to_names.items()})


def _unregister_everything() -> None:
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


def _register_if_absent(collectors: tuple[Collector, ...]) -> None:
    for collector in collectors:
        if collector not in REGISTRY._collector_to_names and not any(
            name in REGISTRY._names_to_collectors for name in REGISTRY._get_names(collector)
        ):
            REGISTRY.register(collector)


def _lazy_owner_collectors() -> tuple[Collector, ...]:
    SpendLogCleanupMetrics._ensure_initialized()
    assert SpendLogCleanupMetrics.rows_deleted is not None
    assert SpendLogCleanupMetrics.batch_duration is not None
    assert SpendLogCleanupMetrics.rows_remaining is not None
    assert SpendLogCleanupMetrics.batch_failures is not None
    assert SpendLogCleanupMetrics.runs is not None
    in_flight: Final = InFlightRequestsMiddleware._get_gauge()
    assert in_flight is not None
    breaker: Final = _breaker_metrics()
    assert breaker._state_gauge is not None
    assert breaker._transitions is not None
    assert breaker._failures is not None
    return (
        SpendLogCleanupMetrics.rows_deleted,
        SpendLogCleanupMetrics.batch_duration,
        SpendLogCleanupMetrics.rows_remaining,
        SpendLogCleanupMetrics.batch_failures,
        SpendLogCleanupMetrics.runs,
        in_flight,
        breaker._state_gauge,
        breaker._transitions,
        breaker._failures,
    )


def _fresh_admission_collectors() -> tuple[Collector, ...]:
    admission: Final = create_prometheus_admission_metrics()
    assert admission is not None
    return (admission.admitted_gauge, admission.queued_gauge, admission.rejected_counter)


@contextmanager
def _isolated_litellm_metric_families(monkeypatch: pytest.MonkeyPatch) -> Iterator[frozenset[str]]:
    previous: Final = _registered_collectors()
    _unregister_everything()
    monkeypatch.setattr(litellm, "prometheus_metrics_config", None)
    PrometheusLogger()
    PrometheusServicesLogger()
    lazy_owned: Final = _lazy_owner_collectors()
    _register_if_absent(lazy_owned)
    _fresh_admission_collectors()
    try:
        yield frozenset(metric.name for metric in REGISTRY.collect())
    finally:
        _unregister_everything()
        for collector in previous:
            REGISTRY.register(collector)
        _register_if_absent(lazy_owned)


@pytest.fixture
def emitted_metric_families(monkeypatch: pytest.MonkeyPatch) -> Iterator[frozenset[str]]:
    with _isolated_litellm_metric_families(monkeypatch) as families:
        yield families


@pytest.fixture
def gauges_registered_by_an_earlier_test() -> Iterator[tuple[Collector, Collector]]:
    sentinel: Final = Gauge("litellm_unrelated_sentinel", "registered by a test outside the isolated block")
    already_registered: Final = REGISTRY._names_to_collectors.get("litellm_admission_admitted_requests")
    admission: Final = already_registered or Gauge(
        "litellm_admission_admitted_requests", "registered directly, bypassing admission_control_state"
    )
    yield (sentinel, admission)
    for gauge in (sentinel,) if already_registered is not None else (sentinel, admission):
        if gauge in REGISTRY._collector_to_names:
            REGISTRY.unregister(gauge)


def test_isolated_metric_families_restore_the_registry_and_keep_lazy_owners_live(
    monkeypatch: pytest.MonkeyPatch, gauges_registered_by_an_earlier_test: tuple[Collector, Collector]
):
    before: Final = _registered_collectors()
    with _isolated_litellm_metric_families(monkeypatch) as families:
        assert "litellm_unrelated_sentinel" not in families
        assert "litellm_admission_admitted_requests" in families
        assert "litellm_in_flight_requests" in families
        assert not any(gauge in REGISTRY._collector_to_names for gauge in gauges_registered_by_an_earlier_test)
    after: Final = _registered_collectors()
    assert all(after[collector] == names for collector, names in before.items())
    lazy_owned: Final = _lazy_owner_collectors()
    assert frozenset(after) - frozenset(before) <= frozenset(lazy_owned)
    assert all(collector in after for collector in lazy_owned)


def _dashboard_expressions(path: Path) -> tuple[str, ...]:
    dashboard: Final = json.loads(path.read_text())
    return tuple(target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", ()))


def _referenced_metric_tokens(path: Path) -> frozenset[str]:
    return frozenset(
        token
        for expr in _dashboard_expressions(path)
        for token in _METRIC_TOKEN_RE.findall(_BY_CLAUSE_RE.sub("", expr))
    )


def _family_of(token: str, families: frozenset[str]) -> str | None:
    candidates: Final = (token.removesuffix(suffix) for suffix in _EXPOSITION_SUFFIXES if token.endswith(suffix))
    return next((candidate for candidate in candidates if candidate in families), None)


def test_all_metrics_dashboard_charts_every_emitted_metric_family(emitted_metric_families: frozenset[str]):
    referenced: Final = _referenced_metric_tokens(_ALL_METRICS_DASHBOARD)
    charted: Final = frozenset(
        family for token in referenced for family in (_family_of(token, emitted_metric_families),) if family
    )
    assert emitted_metric_families - charted == frozenset()


@pytest.mark.parametrize("dashboard_path", _LITELLM_DASHBOARDS, ids=lambda p: p.parent.name)
def test_dashboards_only_reference_emitted_metrics(dashboard_path: Path, emitted_metric_families: frozenset[str]):
    dead: Final = frozenset(
        token
        for token in _referenced_metric_tokens(dashboard_path)
        if _family_of(token, emitted_metric_families) is None
    )
    assert dead == frozenset()


@pytest.mark.parametrize("dashboard_path", _LITELLM_DASHBOARDS, ids=lambda p: p.parent.name)
def test_dashboards_use_templated_prometheus_datasource(dashboard_path: Path):
    dashboard: Final = json.loads(dashboard_path.read_text())
    datasource_variables: Final = tuple(
        variable["name"] for variable in dashboard["templating"]["list"] if variable["type"] == "datasource"
    )
    assert datasource_variables == ("DS_PROMETHEUS",)
    panel_datasource_uids: Final = frozenset(
        panel["datasource"]["uid"] for panel in dashboard["panels"] if panel["type"] != "row"
    )
    assert panel_datasource_uids == frozenset({"${DS_PROMETHEUS}"})


def test_remaining_requests_metric_name_in_defined_metrics():
    """
    Test that litellm_remaining_requests_metric is defined in DEFINED_PROMETHEUS_METRICS.

    The metric name should include the _metric suffix to be consistent with the
    configuration format users specify in prometheus_metrics_config.
    """
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
    assert (
        "litellm_remaining_requests_metric" in defined_metrics
    ), "litellm_remaining_requests_metric should be in DEFINED_PROMETHEUS_METRICS"


def test_remaining_tokens_metric_name_in_defined_metrics():
    """
    Test that litellm_remaining_tokens_metric is defined in DEFINED_PROMETHEUS_METRICS.

    The metric name should include the _metric suffix to be consistent with the
    configuration format users specify in prometheus_metrics_config.
    """
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
    assert (
        "litellm_remaining_tokens_metric" in defined_metrics
    ), "litellm_remaining_tokens_metric should be in DEFINED_PROMETHEUS_METRICS"


def test_prometheus_metric_labels_have_remaining_metrics():
    """
    Test that PrometheusMetricLabels has label definitions for remaining metrics.

    This ensures that the labels can be retrieved when creating the metrics.
    """
    from litellm.types.integrations.prometheus import PrometheusMetricLabels

    # Test that labels can be retrieved for remaining metrics
    remaining_requests_labels = PrometheusMetricLabels.get_labels(
        "litellm_remaining_requests_metric"
    )
    remaining_tokens_labels = PrometheusMetricLabels.get_labels(
        "litellm_remaining_tokens_metric"
    )

    assert isinstance(
        remaining_requests_labels, list
    ), "Labels for litellm_remaining_requests_metric should be a list"
    assert isinstance(
        remaining_tokens_labels, list
    ), "Labels for litellm_remaining_tokens_metric should be a list"

    # These metrics should have api_provider and api_base labels
    assert (
        "api_provider" in remaining_requests_labels
    ), "litellm_remaining_requests_metric should have api_provider label"
    assert (
        "api_base" in remaining_requests_labels
    ), "litellm_remaining_requests_metric should have api_base label"
    assert (
        "api_provider" in remaining_tokens_labels
    ), "litellm_remaining_tokens_metric should have api_provider label"
    assert (
        "api_base" in remaining_tokens_labels
    ), "litellm_remaining_tokens_metric should have api_base label"


def test_all_defined_metrics_have_consistent_naming():
    """
    Test that all metrics defined in DEFINED_PROMETHEUS_METRICS follow
    a consistent naming convention.

    This helps prevent similar inconsistencies in the future.
    """
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)

    for metric_name in defined_metrics:
        # All metrics should start with 'litellm_'
        assert metric_name.startswith(
            "litellm_"
        ), f"Metric {metric_name} should start with 'litellm_'"


if __name__ == "__main__":
    test_remaining_requests_metric_name_in_defined_metrics()
    test_remaining_tokens_metric_name_in_defined_metrics()
    test_prometheus_metric_labels_have_remaining_metrics()
    test_all_defined_metrics_have_consistent_naming()
    print("All prometheus metric name consistency tests passed!")
