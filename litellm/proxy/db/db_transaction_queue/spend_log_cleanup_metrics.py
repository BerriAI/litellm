"""
Prometheus metrics for the spend-log retention cleanup job.

The job runs in the background on a single elected pod, so its cost is invisible
from request-path metrics. These instruments make a run's database footprint
observable: how much it deleted, how long each batch took, how much work is
still outstanding, and why a run stopped.

``prometheus_client`` is an optional dependency, so every recorder degrades to a
no-op when it is absent.
"""

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    # aliased so the annotations below cannot be mistaken for collections.Counter
    from prometheus_client import Counter as PrometheusCounter
    from prometheus_client import Gauge as PrometheusGauge
    from prometheus_client import Histogram as PrometheusHistogram

RunOutcome: TypeAlias = Literal[
    "completed",
    "budget_exhausted",
    "batch_cap_reached",
    "skipped_locked",
    "skipped_disabled",
    "aborted",
]

_BATCH_DURATION_BUCKETS: Final = (0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
_TABLE_LABEL: Final = ("table",)
_OUTCOME_LABEL: Final = ("outcome",)


class SpendLogCleanupMetrics:
    """
    Lazily-registered Prometheus instruments for the retention cleanup job.

    Registration is deferred to first use so that importing this module never
    touches the Prometheus registry, which keeps it safe to import from the
    proxy regardless of whether Prometheus is a configured callback.
    """

    _initialized: bool = False
    rows_deleted: "PrometheusCounter | None" = None
    batch_duration: "PrometheusHistogram | None" = None
    rows_remaining: "PrometheusGauge | None" = None
    batch_failures: "PrometheusCounter | None" = None
    runs: "PrometheusCounter | None" = None

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        try:
            # prometheus_client is an optional extra, so it is resolved here rather
            # than at module import: this module is reachable from proxy startup
            # regardless of whether Prometheus is a configured callback.
            from prometheus_client import Counter, Gauge, Histogram

            cls.rows_deleted = Counter(
                "litellm_spend_log_cleanup_rows_deleted_total",
                "Rows deleted by the spend-log retention cleanup job",
                labelnames=_TABLE_LABEL,
            )
            cls.batch_duration = Histogram(
                "litellm_spend_log_cleanup_batch_duration_seconds",
                "Wall-clock duration of one retention cleanup delete batch",
                labelnames=_TABLE_LABEL,
                buckets=_BATCH_DURATION_BUCKETS,
            )
            cls.rows_remaining = Gauge(
                "litellm_spend_log_cleanup_rows_remaining",
                "Expired rows still awaiting deletion, counted only up to "
                "SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP so the probe itself cannot scan a "
                "large table; a value equal to that cap means at least that many remain",
                labelnames=_TABLE_LABEL,
                multiprocess_mode="livemax",
            )
            cls.batch_failures = Counter(
                "litellm_spend_log_cleanup_batch_failures_total",
                "Retention cleanup delete batches that raised",
                labelnames=_TABLE_LABEL,
            )
            cls.runs = Counter(
                "litellm_spend_log_cleanup_runs_total",
                "Retention cleanup runs, labelled by why the run ended",
                labelnames=_OUTCOME_LABEL,
            )
        except Exception as e:  # noqa: BLE001 - a metrics problem must never fail the cleanup run
            # Covers the extra being absent, a duplicate registration (repeated
            # imports under a test runner), and registry misconfiguration alike.
            verbose_proxy_logger.warning("Could not register spend-log cleanup metrics: %s", e)

    @classmethod
    def record_batch(cls, table_name: str, rows_deleted: int, duration_seconds: float) -> None:
        cls._ensure_initialized()
        if cls.rows_deleted is not None:
            cls.rows_deleted.labels(table=table_name).inc(rows_deleted)
        if cls.batch_duration is not None:
            cls.batch_duration.labels(table=table_name).observe(duration_seconds)

    @classmethod
    def record_batch_failure(cls, table_name: str) -> None:
        cls._ensure_initialized()
        if cls.batch_failures is not None:
            cls.batch_failures.labels(table=table_name).inc()

    @classmethod
    def set_rows_remaining(cls, table_name: str, remaining: int) -> None:
        cls._ensure_initialized()
        if cls.rows_remaining is not None:
            cls.rows_remaining.labels(table=table_name).set(remaining)

    @classmethod
    def record_run(cls, outcome: RunOutcome) -> None:
        cls._ensure_initialized()
        if cls.runs is not None:
            cls.runs.labels(outcome=outcome).inc()
