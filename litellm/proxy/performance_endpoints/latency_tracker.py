import statistics
import threading
from collections import deque
from typing import Dict, List, Optional

from litellm.constants import LITELLM_DETAILED_TIMING, PERF_TRACKER_RING_BUFFER_SIZE

_HISTOGRAM_BUCKETS = [
    (0, 5, "0-5"),
    (5, 10, "5-10"),
    (10, 25, "10-25"),
    (25, 50, "25-50"),
    (50, 100, "50-100"),
    (100, 200, "100-200"),
    (200, 500, "200-500"),
    (500, float("inf"), "500+"),
]


def _compute_histogram(values: List[float]) -> List[Dict]:
    """Returns bucket counts for an overhead latency histogram (ms)."""
    counts = {label: 0 for _, _, label in _HISTOGRAM_BUCKETS}
    for v in values:
        for lo, hi, label in _HISTOGRAM_BUCKETS:
            if lo <= v < hi:
                counts[label] += 1
                break
    return [{"bucket": label, "count": counts[label]} for _, _, label in _HISTOGRAM_BUCKETS]


class _LatencyRingBuffer:
    """Thread-safe ring buffer storing timing samples from recent requests."""

    def __init__(self, maxlen: int = PERF_TRACKER_RING_BUFFER_SIZE) -> None:
        self._lock = threading.Lock()
        self._overhead_ms: deque = deque(maxlen=maxlen)
        self._llm_api_ms: deque = deque(maxlen=maxlen)
        self._pre_processing_ms: deque = deque(maxlen=maxlen)
        self._post_processing_ms: deque = deque(maxlen=maxlen)
        self._total_ms: deque = deque(maxlen=maxlen)

    def record(
        self,
        overhead_ms: Optional[float],
        llm_api_ms: Optional[float],
        pre_processing_ms: Optional[float],
        post_processing_ms: Optional[float],
        total_ms: Optional[float],
    ) -> None:
        with self._lock:
            if overhead_ms is not None:
                self._overhead_ms.append(overhead_ms)
            if llm_api_ms is not None:
                self._llm_api_ms.append(llm_api_ms)
            if pre_processing_ms is not None:
                self._pre_processing_ms.append(pre_processing_ms)
            if post_processing_ms is not None:
                self._post_processing_ms.append(post_processing_ms)
            if total_ms is not None:
                self._total_ms.append(total_ms)

    def stats(self) -> Dict:
        with self._lock:
            overhead_list = list(self._overhead_ms)
            return {
                "overhead": _compute_stats(overhead_list),
                "llm_api": _compute_stats(list(self._llm_api_ms)),
                "pre_processing": _compute_stats(list(self._pre_processing_ms)),
                "post_processing": _compute_stats(list(self._post_processing_ms)),
                "total": _compute_stats(list(self._total_ms)),
                "sample_count": len(self._overhead_ms),
                "overhead_histogram": _compute_histogram(overhead_list),
            }


class _PerModelTracker:
    """Thread-safe per-model latency tracker using one ring buffer per model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, _LatencyRingBuffer] = {}

    def record(
        self,
        model: str,
        overhead_ms: Optional[float],
        llm_api_ms: Optional[float],
        total_ms: Optional[float],
    ) -> None:
        with self._lock:
            if model not in self._models:
                self._models[model] = _LatencyRingBuffer()
            buf = self._models[model]
        # record outside the outer lock to minimise contention
        buf.record(
            overhead_ms=overhead_ms,
            llm_api_ms=llm_api_ms,
            pre_processing_ms=None,
            post_processing_ms=None,
            total_ms=total_ms,
        )

    def stats(self) -> List[Dict]:
        with self._lock:
            snapshot = dict(self._models)

        rows = []
        for model, buf in snapshot.items():
            s = buf.stats()
            if s["sample_count"] == 0:
                continue
            rows.append(
                {
                    "model": model,
                    "overhead": s["overhead"],
                    "llm_api": s["llm_api"],
                    "total": s["total"],
                    "sample_count": s["sample_count"],
                }
            )
        # sort by overhead avg descending so worst offenders appear first
        rows.sort(
            key=lambda r: (r["overhead"] or {}).get("avg_ms", 0),
            reverse=True,
        )
        return rows


def _compute_stats(values: List[float]) -> Optional[Dict]:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "avg_ms": round(statistics.mean(values), 1),
        "p50_ms": round(sorted_vals[n // 2], 1),
        "p95_ms": round(sorted_vals[min(int(n * 0.95), n - 1)], 1),
    }


def _to_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


# Module-level singletons — one per worker process
latency_tracker = _LatencyRingBuffer()
per_model_tracker = _PerModelTracker()


def record_request_timing(hidden_params: dict, model: Optional[str] = None) -> None:
    """
    Call once per completed request with the response's hidden_params.
    Extracts timing fields and appends to the ring buffers (global + per-model).
    """
    total_ms = _to_float(hidden_params.get("_response_ms"))
    overhead_ms = _to_float(hidden_params.get("litellm_overhead_time_ms"))

    # Derive llm_api_ms: prefer explicit timing, fall back to total - overhead
    llm_api_ms = _to_float(hidden_params.get("timing_llm_api_ms"))
    if llm_api_ms is None and total_ms is not None and overhead_ms is not None:
        llm_api_ms = max(0.0, total_ms - overhead_ms)

    latency_tracker.record(
        overhead_ms=overhead_ms,
        llm_api_ms=llm_api_ms,
        pre_processing_ms=_to_float(hidden_params.get("timing_pre_processing_ms"))
        if LITELLM_DETAILED_TIMING
        else None,
        post_processing_ms=_to_float(hidden_params.get("timing_post_processing_ms"))
        if LITELLM_DETAILED_TIMING
        else None,
        total_ms=total_ms,
    )

    if model:
        per_model_tracker.record(
            model=model,
            overhead_ms=overhead_ms,
            llm_api_ms=llm_api_ms,
            total_ms=total_ms,
        )
