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
from litellm.proxy.middleware.admission_control_middleware import admission_control_state
from litellm.proxy.middleware.in_flight_requests_middleware import InFlightRequestsMiddleware

_GRAFANA_DIR: Final = Path(__file__).parents[3] / "cookbook" / "litellm_proxy_server" / "grafana_dashboard"
_ALL_METRICS_DASHBOARD: Final = _GRAFANA_DIR / "dashboard_all_metrics" / "grafana_dashboard.json"
_LITELLM_DASHBOARDS: Final = (_ALL_METRICS_DASHBOARD, _GRAFANA_DIR / "dashboard_v2" / "grafana_dashboard.json")
_METRIC_TOKEN_RE: Final = re.compile(r"\blitellm_[a-z0-9_]+")
_BY_CLAUSE_RE: Final = re.compile(r"\bby\s*\([^)]*\)")
_EXPOSITION_SUFFIXES: Final = ("", "_total", "_bucket", "_sum", "_count", "_created")


def _registered_collectors() -> MappingProxyType[Collector, tuple[str, ...]]:
    return MappingProxyType({collector: tuple(names) for collector, names in REGISTRY._collector_to_names.items()})


def _clear_default_registry_and_lazy_owners() -> None:
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    SpendLogCleanupMetrics._initialized = False
    InFlightRequestsMiddleware._gauge_init_attempted = False
    InFlightRequestsMiddleware._gauge = None
    _breaker_metrics.cache_clear()
    admission_control_state._metrics_init_attempted = False
    admission_control_state._metrics = None


def _lazy_owner_collectors() -> tuple[Collector, ...]:
    SpendLogCleanupMetrics._ensure_initialized()
    assert SpendLogCleanupMetrics.runs is not None
    in_flight: Final = InFlightRequestsMiddleware._get_gauge()
    assert in_flight is not None
    admission: Final = admission_control_state._get_metrics()
    assert admission is not None
    return (SpendLogCleanupMetrics.runs, in_flight, _breaker_metrics()._state_gauge, admission.admitted_gauge)


@contextmanager
def _isolated_litellm_metric_families(monkeypatch: pytest.MonkeyPatch) -> Iterator[frozenset[str]]:
    previous: Final = _registered_collectors()
    _clear_default_registry_and_lazy_owners()
    monkeypatch.setattr(litellm, "prometheus_metrics_config", None)
    PrometheusLogger()
    PrometheusServicesLogger()
    logger_collectors: Final = frozenset(REGISTRY._collector_to_names)
    _lazy_owner_collectors()
    lazy_owner_names: Final = frozenset(
        name
        for collector, names in _registered_collectors().items()
        if collector not in logger_collectors
        for name in names
    )
    try:
        yield frozenset(metric.name for metric in REGISTRY.collect())
    finally:
        _clear_default_registry_and_lazy_owners()
        for collector, names in previous.items():
            if lazy_owner_names.isdisjoint(names):
                REGISTRY.register(collector)


@pytest.fixture
def emitted_metric_families(monkeypatch: pytest.MonkeyPatch) -> Iterator[frozenset[str]]:
    with _isolated_litellm_metric_families(monkeypatch) as families:
        yield families


@pytest.fixture
def unrelated_gauge() -> Iterator[Gauge]:
    gauge: Final = Gauge("litellm_unrelated_sentinel", "registered by a test outside the isolated block")
    yield gauge
    if gauge in REGISTRY._collector_to_names:
        REGISTRY.unregister(gauge)


def test_isolated_metric_families_restore_unrelated_collectors_and_lazy_owners(
    monkeypatch: pytest.MonkeyPatch, unrelated_gauge: Gauge
):
    stale: Final = _lazy_owner_collectors()
    with _isolated_litellm_metric_families(monkeypatch) as families:
        assert "litellm_unrelated_sentinel" not in families
        assert unrelated_gauge not in REGISTRY._collector_to_names
    assert unrelated_gauge in REGISTRY._collector_to_names
    assert all(collector not in REGISTRY._collector_to_names for collector in stale)
    assert all(collector in REGISTRY._collector_to_names for collector in _lazy_owner_collectors())


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
