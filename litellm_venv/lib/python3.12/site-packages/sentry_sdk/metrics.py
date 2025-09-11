import io
import os
import random
import re
import sys
import threading
import time
import warnings
import zlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps, partial

import sentry_sdk
from sentry_sdk.utils import (
    ContextVar,
    now,
    nanosecond_time,
    to_timestamp,
    serialize_frame,
    json_dumps,
)
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
    TRANSACTION_SOURCE_ROUTE,
    TRANSACTION_SOURCE_VIEW,
    TRANSACTION_SOURCE_COMPONENT,
    TRANSACTION_SOURCE_TASK,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import Generator
    from typing import Iterable
    from typing import List
    from typing import Optional
    from typing import Set
    from typing import Tuple
    from typing import Union

    from sentry_sdk._types import BucketKey
    from sentry_sdk._types import DurationUnit
    from sentry_sdk._types import FlushedMetricValue
    from sentry_sdk._types import MeasurementUnit
    from sentry_sdk._types import MetricMetaKey
    from sentry_sdk._types import MetricTagValue
    from sentry_sdk._types import MetricTags
    from sentry_sdk._types import MetricTagsInternal
    from sentry_sdk._types import MetricType
    from sentry_sdk._types import MetricValue


warnings.warn(
    "The sentry_sdk.metrics module is deprecated and will be removed in the next major release. "
    "Sentry will reject all metrics sent after October 7, 2024. "
    "Learn more: https://sentry.zendesk.com/hc/en-us/articles/26369339769883-Upcoming-API-Changes-to-Metrics",
    DeprecationWarning,
    stacklevel=2,
)

_in_metrics = ContextVar("in_metrics", default=False)
_set = set  # set is shadowed below

GOOD_TRANSACTION_SOURCES = frozenset(
    [
        TRANSACTION_SOURCE_ROUTE,
        TRANSACTION_SOURCE_VIEW,
        TRANSACTION_SOURCE_COMPONENT,
        TRANSACTION_SOURCE_TASK,
    ]
)

_sanitize_unit = partial(re.compile(r"[^a-zA-Z0-9_]+").sub, "")
_sanitize_metric_key = partial(re.compile(r"[^a-zA-Z0-9_\-.]+").sub, "_")
_sanitize_tag_key = partial(re.compile(r"[^a-zA-Z0-9_\-.\/]+").sub, "")


def _sanitize_tag_value(value):
    # type: (str) -> str
    table = str.maketrans(
        {
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            "\\": "\\\\",
            "|": "\\u{7c}",
            ",": "\\u{2c}",
        }
    )
    return value.translate(table)


def get_code_location(stacklevel):
    # type: (int) -> Optional[Dict[str, Any]]
    try:
        frm = sys._getframe(stacklevel)
    except Exception:
        return None

    return serialize_frame(
        frm, include_local_variables=False, include_source_context=True
    )


@contextmanager
def recursion_protection():
    # type: () -> Generator[bool, None, None]
    """Enters recursion protection and returns the old flag."""
    old_in_metrics = _in_metrics.get()
    _in_metrics.set(True)
    try:
        yield old_in_metrics
    finally:
        _in_metrics.set(old_in_metrics)


def metrics_noop(func):
    # type: (Any) -> Any
    """Convenient decorator that uses `recursion_protection` to
    make a function a noop.
    """

    @wraps(func)
    def new_func(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        with recursion_protection() as in_metrics:
            if not in_metrics:
                return func(*args, **kwargs)

    return new_func


class Metric(ABC):
    __slots__ = ()

    @abstractmethod
    def __init__(self, first):
        # type: (MetricValue) -> None
        pass

    @property
    @abstractmethod
    def weight(self):
        # type: () -> int
        pass

    @abstractmethod
    def add(self, value):
        # type: (MetricValue) -> None
        pass

    @abstractmethod
    def serialize_value(self):
        # type: () -> Iterable[FlushedMetricValue]
        pass


class CounterMetric(Metric):
    __slots__ = ("value",)

    def __init__(
        self, first  # type: MetricValue
    ):
        # type: (...) -> None
        self.value = float(first)

    @property
    def weight(self):
        # type: (...) -> int
        return 1

    def add(
        self, value  # type: MetricValue
    ):
        # type: (...) -> None
        self.value += float(value)

    def serialize_value(self):
        # type: (...) -> Iterable[FlushedMetricValue]
        return (self.value,)


class GaugeMetric(Metric):
    __slots__ = (
        "last",
        "min",
        "max",
        "sum",
        "count",
    )

    def __init__(
        self, first  # type: MetricValue
    ):
        # type: (...) -> None
        first = float(first)
        self.last = first
        self.min = first
        self.max = first
        self.sum = first
        self.count = 1

    @property
    def weight(self):
        # type: (...) -> int
        # Number of elements.
        return 5

    def add(
        self, value  # type: MetricValue
    ):
        # type: (...) -> None
        value = float(value)
        self.last = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.sum += value
        self.count += 1

    def serialize_value(self):
        # type: (...) -> Iterable[FlushedMetricValue]
        return (
            self.last,
            self.min,
            self.max,
            self.sum,
            self.count,
        )


class DistributionMetric(Metric):
    __slots__ = ("value",)

    def __init__(
        self, first  # type: MetricValue
    ):
        # type(...) -> None
        self.value = [float(first)]

    @property
    def weight(self):
        # type: (...) -> int
        return len(self.value)

    def add(
        self, value  # type: MetricValue
    ):
        # type: (...) -> None
        self.value.append(float(value))

    def serialize_value(self):
        # type: (...) -> Iterable[FlushedMetricValue]
        return self.value


class SetMetric(Metric):
    __slots__ = ("value",)

    def __init__(
        self, first  # type: MetricValue
    ):
        # type: (...) -> None
        self.value = {first}

    @property
    def weight(self):
        # type: (...) -> int
        return len(self.value)

    def add(
        self, value  # type: MetricValue
    ):
        # type: (...) -> None
        self.value.add(value)

    def serialize_value(self):
        # type: (...) -> Iterable[FlushedMetricValue]
        def _hash(x):
            # type: (MetricValue) -> int
            if isinstance(x, str):
                return zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF
            return int(x)

        return (_hash(value) for value in self.value)


def _encode_metrics(flushable_buckets):
    # type: (Iterable[Tuple[int, Dict[BucketKey, Metric]]]) -> bytes
    out = io.BytesIO()
    _write = out.write

    # Note on sanitization: we intentionally sanitize in emission (serialization)
    # and not during aggregation for performance reasons.  This means that the
    # envelope can in fact have duplicate buckets stored.  This is acceptable for
    # relay side emission and should not happen commonly.

    for timestamp, buckets in flushable_buckets:
        for bucket_key, metric in buckets.items():
            metric_type, metric_name, metric_unit, metric_tags = bucket_key
            metric_name = _sanitize_metric_key(metric_name)
            metric_unit = _sanitize_unit(metric_unit)
            _write(metric_name.encode("utf-8"))
            _write(b"@")
            _write(metric_unit.encode("utf-8"))

            for serialized_value in metric.serialize_value():
                _write(b":")
                _write(str(serialized_value).encode("utf-8"))

            _write(b"|")
            _write(metric_type.encode("ascii"))

            if metric_tags:
                _write(b"|#")
                first = True
                for tag_key, tag_value in metric_tags:
                    tag_key = _sanitize_tag_key(tag_key)
                    if not tag_key:
                        continue
                    if first:
                        first = False
                    else:
                        _write(b",")
                    _write(tag_key.encode("utf-8"))
                    _write(b":")
                    _write(_sanitize_tag_value(tag_value).encode("utf-8"))

            _write(b"|T")
            _write(str(timestamp).encode("ascii"))
            _write(b"\n")

    return out.getvalue()


def _encode_locations(timestamp, code_locations):
    # type: (int, Iterable[Tuple[MetricMetaKey, Dict[str, Any]]]) -> bytes
    mapping = {}  # type: Dict[str, List[Any]]

    for key, loc in code_locations:
        metric_type, name, unit = key
        mri = "{}:{}@{}".format(
            metric_type, _sanitize_metric_key(name), _sanitize_unit(unit)
        )

        loc["type"] = "location"
        mapping.setdefault(mri, []).append(loc)

    return json_dumps({"timestamp": timestamp, "mapping": mapping})


METRIC_TYPES = {
    "c": CounterMetric,
    "g": GaugeMetric,
    "d": DistributionMetric,
    "s": SetMetric,
}  # type: dict[MetricType, type[Metric]]

# some of these are dumb
TIMING_FUNCTIONS = {
    "nanosecond": nanosecond_time,
    "microsecond": lambda: nanosecond_time() / 1000.0,
    "millisecond": lambda: nanosecond_time() / 1000000.0,
    "second": now,
    "minute": lambda: now() / 60.0,
    "hour": lambda: now() / 3600.0,
    "day": lambda: now() / 3600.0 / 24.0,
    "week": lambda: now() / 3600.0 / 24.0 / 7.0,
}


class LocalAggregator:
    __slots__ = ("_measurements",)

    def __init__(self):
        # type: (...) -> None
        self._measurements = (
            {}
        )  # type: Dict[Tuple[str, MetricTagsInternal], Tuple[float, float, int, float]]

    def add(
        self,
        ty,  # type: MetricType
        key,  # type: str
        value,  # type: float
        unit,  # type: MeasurementUnit
        tags,  # type: MetricTagsInternal
    ):
        # type: (...) -> None
        export_key = "%s:%s@%s" % (ty, key, unit)
        bucket_key = (export_key, tags)

        old = self._measurements.get(bucket_key)
        if old is not None:
            v_min, v_max, v_count, v_sum = old
            v_min = min(v_min, value)
            v_max = max(v_max, value)
            v_count += 1
            v_sum += value
        else:
            v_min = v_max = v_sum = value
            v_count = 1
        self._measurements[bucket_key] = (v_min, v_max, v_count, v_sum)

    def to_json(self):
        # type: (...) -> Dict[str, Any]
        rv = {}  # type: Any
        for (export_key, tags), (
            v_min,
            v_max,
            v_count,
            v_sum,
        ) in self._measurements.items():
            rv.setdefault(export_key, []).append(
                {
                    "tags": _tags_to_dict(tags),
                    "min": v_min,
                    "max": v_max,
                    "count": v_count,
                    "sum": v_sum,
                }
            )
        return rv


class MetricsAggregator:
    ROLLUP_IN_SECONDS = 10.0
    MAX_WEIGHT = 100000
    FLUSHER_SLEEP_TIME = 5.0

    def __init__(
        self,
        capture_func,  # type: Callable[[Envelope], None]
        enable_code_locations=False,  # type: bool
    ):
        # type: (...) -> None
        self.buckets = {}  # type: Dict[int, Any]
        self._enable_code_locations = enable_code_locations
        self._seen_locations = _set()  # type: Set[Tuple[int, MetricMetaKey]]
        self._pending_locations = {}  # type: Dict[int, List[Tuple[MetricMetaKey, Any]]]
        self._buckets_total_weight = 0
        self._capture_func = capture_func
        self._running = True
        self._lock = threading.Lock()

        self._flush_event = threading.Event()  # type: threading.Event
        self._force_flush = False

        # The aggregator shifts its flushing by up to an entire rollup window to
        # avoid multiple clients trampling on end of a 10 second window as all the
        # buckets are anchored to multiples of ROLLUP seconds.  We randomize this
        # number once per aggregator boot to achieve some level of offsetting
        # across a fleet of deployed SDKs.  Relay itself will also apply independent
        # jittering.
        self._flush_shift = random.random() * self.ROLLUP_IN_SECONDS

        self._flusher = None  # type: Optional[threading.Thread]
        self._flusher_pid = None  # type: Optional[int]

    def _ensure_thread(self):
        # type: (...) -> bool
        """For forking processes we might need to restart this thread.
        This ensures that our process actually has that thread running.
        """
        if not self._running:
            return False

        pid = os.getpid()
        if self._flusher_pid == pid:
            return True

        with self._lock:
            # Recheck to make sure another thread didn't get here and start the
            # the flusher in the meantime
            if self._flusher_pid == pid:
                return True

            self._flusher_pid = pid

            self._flusher = threading.Thread(target=self._flush_loop)
            self._flusher.daemon = True

            try:
                self._flusher.start()
            except RuntimeError:
                # Unfortunately at this point the interpreter is in a state that no
                # longer allows us to spawn a thread and we have to bail.
                self._running = False
                return False

        return True

    def _flush_loop(self):
        # type: (...) -> None
        _in_metrics.set(True)
        while self._running or self._force_flush:
            if self._running:
                self._flush_event.wait(self.FLUSHER_SLEEP_TIME)
            self._flush()

    def _flush(self):
        # type: (...) -> None
        self._emit(self._flushable_buckets(), self._flushable_locations())

    def _flushable_buckets(self):
        # type: (...) -> (Iterable[Tuple[int, Dict[BucketKey, Metric]]])
        with self._lock:
            force_flush = self._force_flush
            cutoff = time.time() - self.ROLLUP_IN_SECONDS - self._flush_shift
            flushable_buckets = ()  # type: Iterable[Tuple[int, Dict[BucketKey, Metric]]]
            weight_to_remove = 0

            if force_flush:
                flushable_buckets = self.buckets.items()
                self.buckets = {}
                self._buckets_total_weight = 0
                self._force_flush = False
            else:
                flushable_buckets = []
                for buckets_timestamp, buckets in self.buckets.items():
                    # If the timestamp of the bucket is newer that the rollup we want to skip it.
                    if buckets_timestamp <= cutoff:
                        flushable_buckets.append((buckets_timestamp, buckets))

                # We will clear the elements while holding the lock, in order to avoid requesting it downstream again.
                for buckets_timestamp, buckets in flushable_buckets:
                    for metric in buckets.values():
                        weight_to_remove += metric.weight
                    del self.buckets[buckets_timestamp]

                self._buckets_total_weight -= weight_to_remove

        return flushable_buckets

    def _flushable_locations(self):
        # type: (...) -> Dict[int, List[Tuple[MetricMetaKey, Dict[str, Any]]]]
        with self._lock:
            locations = self._pending_locations
            self._pending_locations = {}
        return locations

    @metrics_noop
    def add(
        self,
        ty,  # type: MetricType
        key,  # type: str
        value,  # type: MetricValue
        unit,  # type: MeasurementUnit
        tags,  # type: Optional[MetricTags]
        timestamp=None,  # type: Optional[Union[float, datetime]]
        local_aggregator=None,  # type: Optional[LocalAggregator]
        stacklevel=0,  # type: Optional[int]
    ):
        # type: (...) -> None
        if not self._ensure_thread() or self._flusher is None:
            return None

        if timestamp is None:
            timestamp = time.time()
        elif isinstance(timestamp, datetime):
            timestamp = to_timestamp(timestamp)

        bucket_timestamp = int(
            (timestamp // self.ROLLUP_IN_SECONDS) * self.ROLLUP_IN_SECONDS
        )
        serialized_tags = _serialize_tags(tags)
        bucket_key = (
            ty,
            key,
            unit,
            serialized_tags,
        )

        with self._lock:
            local_buckets = self.buckets.setdefault(bucket_timestamp, {})
            metric = local_buckets.get(bucket_key)
            if metric is not None:
                previous_weight = metric.weight
                metric.add(value)
            else:
                metric = local_buckets[bucket_key] = METRIC_TYPES[ty](value)
                previous_weight = 0

            added = metric.weight - previous_weight

            if stacklevel is not None:
                self.record_code_location(ty, key, unit, stacklevel + 2, timestamp)

        # Given the new weight we consider whether we want to force flush.
        self._consider_force_flush()

        # For sets, we only record that a value has been added to the set but not which one.
        # See develop docs: https://develop.sentry.dev/sdk/metrics/#sets
        if local_aggregator is not None:
            local_value = float(added if ty == "s" else value)
            local_aggregator.add(ty, key, local_value, unit, serialized_tags)

    def record_code_location(
        self,
        ty,  # type: MetricType
        key,  # type: str
        unit,  # type: MeasurementUnit
        stacklevel,  # type: int
        timestamp=None,  # type: Optional[float]
    ):
        # type: (...) -> None
        if not self._enable_code_locations:
            return
        if timestamp is None:
            timestamp = time.time()
        meta_key = (ty, key, unit)
        start_of_day = datetime.fromtimestamp(timestamp, timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        start_of_day = int(to_timestamp(start_of_day))

        if (start_of_day, meta_key) not in self._seen_locations:
            self._seen_locations.add((start_of_day, meta_key))
            loc = get_code_location(stacklevel + 3)
            if loc is not None:
                # Group metadata by day to make flushing more efficient.
                # There needs to be one envelope item per timestamp.
                self._pending_locations.setdefault(start_of_day, []).append(
                    (meta_key, loc)
                )

    @metrics_noop
    def need_code_location(
        self,
        ty,  # type: MetricType
        key,  # type: str
        unit,  # type: MeasurementUnit
        timestamp,  # type: float
    ):
        # type: (...) -> bool
        if self._enable_code_locations:
            return False
        meta_key = (ty, key, unit)
        start_of_day = datetime.fromtimestamp(timestamp, timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        start_of_day = int(to_timestamp(start_of_day))
        return (start_of_day, meta_key) not in self._seen_locations

    def kill(self):
        # type: (...) -> None
        if self._flusher is None:
            return

        self._running = False
        self._flush_event.set()
        self._flusher = None

    @metrics_noop
    def flush(self):
        # type: (...) -> None
        self._force_flush = True
        self._flush()

    def _consider_force_flush(self):
        # type: (...) -> None
        # It's important to acquire a lock around this method, since it will touch shared data structures.
        total_weight = len(self.buckets) + self._buckets_total_weight
        if total_weight >= self.MAX_WEIGHT:
            self._force_flush = True
            self._flush_event.set()

    def _emit(
        self,
        flushable_buckets,  # type: (Iterable[Tuple[int, Dict[BucketKey, Metric]]])
        code_locations,  # type: Dict[int, List[Tuple[MetricMetaKey, Dict[str, Any]]]]
    ):
        # type: (...) -> Optional[Envelope]
        envelope = Envelope()

        if flushable_buckets:
            encoded_metrics = _encode_metrics(flushable_buckets)
            envelope.add_item(Item(payload=encoded_metrics, type="statsd"))

        for timestamp, locations in code_locations.items():
            encoded_locations = _encode_locations(timestamp, locations)
            envelope.add_item(Item(payload=encoded_locations, type="metric_meta"))

        if envelope.items:
            self._capture_func(envelope)
            return envelope
        return None


def _serialize_tags(
    tags,  # type: Optional[MetricTags]
):
    # type: (...) -> MetricTagsInternal
    if not tags:
        return ()

    rv = []
    for key, value in tags.items():
        # If the value is a collection, we want to flatten it.
        if isinstance(value, (list, tuple)):
            for inner_value in value:
                if inner_value is not None:
                    rv.append((key, str(inner_value)))
        elif value is not None:
            rv.append((key, str(value)))

    # It's very important to sort the tags in order to obtain the
    # same bucket key.
    return tuple(sorted(rv))


def _tags_to_dict(tags):
    # type: (MetricTagsInternal) -> Dict[str, Any]
    rv = {}  # type: Dict[str, Any]
    for tag_name, tag_value in tags:
        old_value = rv.get(tag_name)
        if old_value is not None:
            if isinstance(old_value, list):
                old_value.append(tag_value)
            else:
                rv[tag_name] = [old_value, tag_value]
        else:
            rv[tag_name] = tag_value
    return rv


def _get_aggregator():
    # type: () -> Optional[MetricsAggregator]
    client = sentry_sdk.get_client()
    return (
        client.metrics_aggregator
        if client.is_active() and client.metrics_aggregator is not None
        else None
    )


def _get_aggregator_and_update_tags(key, value, unit, tags):
    # type: (str, Optional[MetricValue], MeasurementUnit, Optional[MetricTags]) -> Tuple[Optional[MetricsAggregator], Optional[LocalAggregator], Optional[MetricTags]]
    client = sentry_sdk.get_client()
    if not client.is_active() or client.metrics_aggregator is None:
        return None, None, tags

    updated_tags = dict(tags or ())  # type: Dict[str, MetricTagValue]
    updated_tags.setdefault("release", client.options["release"])
    updated_tags.setdefault("environment", client.options["environment"])

    scope = sentry_sdk.get_current_scope()
    local_aggregator = None

    # We go with the low-level API here to access transaction information as
    # this one is the same between just errors and errors + performance
    transaction_source = scope._transaction_info.get("source")
    if transaction_source in GOOD_TRANSACTION_SOURCES:
        transaction_name = scope._transaction
        if transaction_name:
            updated_tags.setdefault("transaction", transaction_name)
        if scope._span is not None:
            local_aggregator = scope._span._get_local_aggregator()

    experiments = client.options.get("_experiments", {})
    before_emit_callback = experiments.get("before_emit_metric")
    if before_emit_callback is not None:
        with recursion_protection() as in_metrics:
            if not in_metrics:
                if not before_emit_callback(key, value, unit, updated_tags):
                    return None, None, updated_tags

    return client.metrics_aggregator, local_aggregator, updated_tags


def increment(
    key,  # type: str
    value=1.0,  # type: float
    unit="none",  # type: MeasurementUnit
    tags=None,  # type: Optional[MetricTags]
    timestamp=None,  # type: Optional[Union[float, datetime]]
    stacklevel=0,  # type: int
):
    # type: (...) -> None
    """Increments a counter."""
    aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(
        key, value, unit, tags
    )
    if aggregator is not None:
        aggregator.add(
            "c", key, value, unit, tags, timestamp, local_aggregator, stacklevel
        )


# alias as incr is relatively common in python
incr = increment


class _Timing:
    def __init__(
        self,
        key,  # type: str
        tags,  # type: Optional[MetricTags]
        timestamp,  # type: Optional[Union[float, datetime]]
        value,  # type: Optional[float]
        unit,  # type: DurationUnit
        stacklevel,  # type: int
    ):
        # type: (...) -> None
        self.key = key
        self.tags = tags
        self.timestamp = timestamp
        self.value = value
        self.unit = unit
        self.entered = None  # type: Optional[float]
        self._span = None  # type: Optional[sentry_sdk.tracing.Span]
        self.stacklevel = stacklevel

    def _validate_invocation(self, context):
        # type: (str) -> None
        if self.value is not None:
            raise TypeError(
                "cannot use timing as %s when a value is provided" % context
            )

    def __enter__(self):
        # type: (...) -> _Timing
        self.entered = TIMING_FUNCTIONS[self.unit]()
        self._validate_invocation("context-manager")
        self._span = sentry_sdk.start_span(op="metric.timing", name=self.key)
        if self.tags:
            for key, value in self.tags.items():
                if isinstance(value, (tuple, list)):
                    value = ",".join(sorted(map(str, value)))
                self._span.set_tag(key, value)
        self._span.__enter__()

        # report code locations here for better accuracy
        aggregator = _get_aggregator()
        if aggregator is not None:
            aggregator.record_code_location("d", self.key, self.unit, self.stacklevel)

        return self

    def __exit__(self, exc_type, exc_value, tb):
        # type: (Any, Any, Any) -> None
        assert self._span, "did not enter"
        aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(
            self.key,
            self.value,
            self.unit,
            self.tags,
        )
        if aggregator is not None:
            elapsed = TIMING_FUNCTIONS[self.unit]() - self.entered  # type: ignore
            aggregator.add(
                "d",
                self.key,
                elapsed,
                self.unit,
                tags,
                self.timestamp,
                local_aggregator,
                None,  # code locations are reported in __enter__
            )

        self._span.__exit__(exc_type, exc_value, tb)
        self._span = None

    def __call__(self, f):
        # type: (Any) -> Any
        self._validate_invocation("decorator")

        @wraps(f)
        def timed_func(*args, **kwargs):
            # type: (*Any, **Any) -> Any
            with timing(
                key=self.key,
                tags=self.tags,
                timestamp=self.timestamp,
                unit=self.unit,
                stacklevel=self.stacklevel + 1,
            ):
                return f(*args, **kwargs)

        return timed_func


def timing(
    key,  # type: str
    value=None,  # type: Optional[float]
    unit="second",  # type: DurationUnit
    tags=None,  # type: Optional[MetricTags]
    timestamp=None,  # type: Optional[Union[float, datetime]]
    stacklevel=0,  # type: int
):
    # type: (...) -> _Timing
    """Emits a distribution with the time it takes to run the given code block.

    This method supports three forms of invocation:

    - when a `value` is provided, it functions similar to `distribution` but with
    - it can be used as a context manager
    - it can be used as a decorator
    """
    if value is not None:
        aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(
            key, value, unit, tags
        )
        if aggregator is not None:
            aggregator.add(
                "d", key, value, unit, tags, timestamp, local_aggregator, stacklevel
            )
    return _Timing(key, tags, timestamp, value, unit, stacklevel)


def distribution(
    key,  # type: str
    value,  # type: float
    unit="none",  # type: MeasurementUnit
    tags=None,  # type: Optional[MetricTags]
    timestamp=None,  # type: Optional[Union[float, datetime]]
    stacklevel=0,  # type: int
):
    # type: (...) -> None
    """Emits a distribution."""
    aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(
        key, value, unit, tags
    )
    if aggregator is not None:
        aggregator.add(
            "d", key, value, unit, tags, timestamp, local_aggregator, stacklevel
        )


def set(
    key,  # type: str
    value,  # type: Union[int, str]
    unit="none",  # type: MeasurementUnit
    tags=None,  # type: Optional[MetricTags]
    timestamp=None,  # type: Optional[Union[float, datetime]]
    stacklevel=0,  # type: int
):
    # type: (...) -> None
    """Emits a set."""
    aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(
        key, value, unit, tags
    )
    if aggregator is not None:
        aggregator.add(
            "s", key, value, unit, tags, timestamp, local_aggregator, stacklevel
        )


def gauge(
    key,  # type: str
    value,  # type: float
    unit="none",  # type: MeasurementUnit
    tags=None,  # type: Optional[MetricTags]
    timestamp=None,  # type: Optional[Union[float, datetime]]
    stacklevel=0,  # type: int
):
    # type: (...) -> None
    """Emits a gauge."""
    aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(
        key, value, unit, tags
    )
    if aggregator is not None:
        aggregator.add(
            "g", key, value, unit, tags, timestamp, local_aggregator, stacklevel
        )
