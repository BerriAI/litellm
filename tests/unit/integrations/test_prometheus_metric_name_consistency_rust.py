import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from prometheus_client import REGISTRY, Gauge
from prometheus_client.registry import Collector

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_services import PrometheusServicesLogger
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup_metrics import SpendLogCleanupMetrics
from litellm.proxy.middleware.admission_control_middleware import create_prometheus_admission_metrics
from litellm.proxy.middleware.in_flight_requests_middleware import InFlightRequestsMiddleware
from litellm.tracing.exporter import lens_export_counters
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


_GRAFANA_DIR: Final = (
    Path(__file__).parents[3] / "litellm" / "integrations" / "prometheus_helpers" / "grafana_dashboards"
)
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
    services: Final = PrometheusServicesLogger()
    lens: Final = lens_export_counters()
    assert lens.written is not None
    assert lens.dropped is not None
    return (
        SpendLogCleanupMetrics.rows_deleted,
        SpendLogCleanupMetrics.batch_duration,
        SpendLogCleanupMetrics.rows_remaining,
        SpendLogCleanupMetrics.batch_failures,
        SpendLogCleanupMetrics.runs,
        in_flight,
        services.redis_breaker_state,
        services.redis_breaker_transitions,
        services.redis_breaker_failures,
        lens.written,
        lens.dropped,
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
