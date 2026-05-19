from __future__ import annotations

import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, Optional


class BoundedPrometheusSeriesTracker:
    """
    Tracks Prometheus child series and removes stale/excess labelsets.

    The tracker is label-agnostic: callers decide which series should be tracked
    and pass the full label tuple used by the Prometheus metric.
    """

    def __init__(self) -> None:
        self._series: Dict[str, OrderedDict[tuple[Optional[str], ...], float]] = {}
        self._last_ttl_cleanup: Dict[str, float] = {}
        self.lock = RLock()

    def track_series(
        self,
        metric: Any,
        metric_name: str,
        label_values: tuple[Optional[str], ...],
        max_series: Optional[int],
        ttl_seconds: Optional[float],
        cleanup_interval_seconds: Optional[float],
    ) -> None:
        if max_series is None and ttl_seconds is None:
            return

        now = time.monotonic()

        with self.lock:
            series = self._series.setdefault(metric_name, OrderedDict())
            series[label_values] = now
            series.move_to_end(label_values)

            if ttl_seconds is not None and self._should_run_ttl_cleanup(
                metric_name=metric_name,
                now=now,
                cleanup_interval_seconds=cleanup_interval_seconds,
            ):
                expired_label_values = [
                    tracked_label_values
                    for tracked_label_values, last_seen in series.items()
                    if now - last_seen > ttl_seconds
                ]
                for tracked_label_values in expired_label_values:
                    self._remove_metric_series(metric, series, tracked_label_values)

            # max_series <= 0 is treated as "unlimited" so a misconfigured zero
            # value cannot silently drop every emission for this metric.
            if max_series is not None and max_series > 0:
                while len(series) > max_series:
                    tracked_label_values = next(iter(series))
                    if not self._remove_metric_child(metric, tracked_label_values):
                        break
                    del series[tracked_label_values]

    def _should_run_ttl_cleanup(
        self,
        metric_name: str,
        now: float,
        cleanup_interval_seconds: Optional[float],
    ) -> bool:
        if cleanup_interval_seconds is None or cleanup_interval_seconds <= 0:
            self._last_ttl_cleanup[metric_name] = now
            return True

        last_cleanup = self._last_ttl_cleanup.get(metric_name)
        if last_cleanup is None or now - last_cleanup >= cleanup_interval_seconds:
            self._last_ttl_cleanup[metric_name] = now
            return True
        return False

    def _remove_metric_series(
        self,
        metric: Any,
        series: OrderedDict[tuple[Optional[str], ...], float],
        label_values: tuple[Optional[str], ...],
    ) -> None:
        if self._remove_metric_child(metric, label_values):
            series.pop(label_values, None)

    @staticmethod
    def _remove_metric_child(
        metric: Any, label_values: tuple[Optional[str], ...]
    ) -> bool:
        """
        Remove the Prometheus child for ``label_values`` and report whether the
        tracker should commit the matching state change.

        Returns ``True`` when the child is no longer present in Prometheus
        (either it was just removed or it was already gone), and ``False`` when
        ``metric.remove()`` raised an unexpected error and the child likely
        still exists.
        """
        try:
            metric.remove(*label_values)
            return True
        except KeyError:
            return True
        except (AttributeError, ValueError):
            return False
