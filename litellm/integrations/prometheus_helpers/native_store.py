from __future__ import annotations

import builtins
import json
import math
import os
import traceback
import weakref
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar, get_args

from prometheus_client import REGISTRY
from prometheus_client import metrics as prometheus_metrics
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import Collector
from prometheus_client.utils import (
    floatToGoString,  # pyright: ignore[reportUnknownVariableType]  # the client ships no annotations
)
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import Admit, KwargsFields, hook_subscribers, run_subscribers
from litellm.integrations.custom_logger_v2.rust import RustLogger, rust_logger
from litellm.integrations.prometheus_helpers.forwarding_store import ForwardingStore, PrometheusKind, forwarded_metric
from litellm.integrations.prometheus_helpers.shared_prometheus_series_admissions import (
    SharedPrometheusSeriesAdmissions,
)
from litellm.litellm_core_utils.core_helpers import env_flag
from litellm.rust_bridge import _native
from litellm.types.integrations.custom_logger_v2 import Outcome
from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS, NoOpMetric, PrometheusMetricLabels

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger

_Parsed: Final = TypeVar("_Parsed")
NativeStep: TypeAlias = Literal["success_head", "success_limits", "success_latency", "success_tail", "failure"]
NativeEvent: TypeAlias = Literal["success", "failure"]
NativeOp: TypeAlias = Literal[
    "deployment_state",
    "deployment_failure",
    "cooled_down",
    "guardrail",
    "managed_batch_created",
    "managed_file_created",
    "managed_file_size",
    "managed_batch_duration",
    "managed_file_deleted",
    "check_batch_cost_run",
    "check_batch_cost_error",
    "callback_failure",
    "provider_budget",
    "spend_capture_rate",
    "dynamic_limiter_share",
    "dynamic_limiter_throttled",
    "fallback",
    "proxy_failure",
    "budget",
    "team_members",
    "counts",
]

NATIVE_PROMETHEUS_ENV: Final = "LITELLM_NATIVE_PROMETHEUS"
NativePrometheusMetric: Final = _native.NativePrometheusMetric

_PAYLOAD_LABEL_KEYS: Final = ("model_group", "model_id", "api_base", "custom_llm_provider")
_EPOCH_NAIVE: Final = datetime(1970, 1, 1)  # noqa: DTZ001  # callback timestamps use the legacy naive contract
_EPOCH_AWARE: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MICROSECOND: Final = timedelta(microseconds=1)
_JSON_INTS: Final = (-(2**63), 2**64 - 1)
NUMBER_KEY: Final = "$num"
_MAPPING: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_MOST_RECENT_MODES: Final = frozenset(("mostrecent", "livemostrecent"))
_STRICT: Final = ConfigDict(strict=True)
_DICT: Final[TypeAdapter[dict[object, object]]] = TypeAdapter(  # mutable-ok: pydantic hands back a fresh dict per call
    dict[object, object], config=_STRICT
)
_LIST: Final[TypeAdapter[list[object]]] = TypeAdapter(  # mutable-ok: pydantic hands back a fresh list per call
    list[object], config=_STRICT
)
_TUPLE: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...], config=_STRICT)
_DEFINED_METRICS: Final = TypeAdapter(tuple[DEFINED_PROMETHEUS_METRICS, ...]).validate_python(
    get_args(DEFINED_PROMETHEUS_METRICS)
)
_KINDS: Final[Mapping[str, PrometheusKind]] = {"counter": "counter", "gauge": "gauge", "histogram": "histogram"}


def native_prometheus_enabled() -> bool:
    return env_flag(os.environ, NATIVE_PROMETHEUS_ENV, verbose_logger)


def _parsed(adapter: TypeAdapter[_Parsed], value: object) -> _Parsed | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


def _items(value: object) -> Sequence[object] | None:
    listed: Final = _parsed(_LIST, value)
    return listed if listed is not None else _parsed(_TUPLE, value)


def json_safe(value: object) -> object:
    root: Final[list[object]] = [None]
    pending: Final[list[tuple[Callable[[object], None], object, frozenset[int]]]] = [
        (partial(root.__setitem__, 0), value, frozenset())
    ]
    while pending:
        assign, current, path = pending.pop()
        converted, children = _json_node(current)
        assign(converted)
        if children:
            if id(current) in path:
                raise ValueError("Circular reference detected")
            nested = path | {id(current)}
            pending.extend((child_assign, child, nested) for child_assign, child in reversed(children))
    return root[0]


def _json_node(value: object) -> tuple[object, Sequence[tuple[Callable[[object], None], object]]]:
    if value is None or isinstance(value, bool | str):
        return value, ()
    if isinstance(value, int):
        return (value if _JSON_INTS[0] <= value <= _JSON_INTS[1] else {NUMBER_KEY: str(value)}), ()
    if isinstance(value, float):
        return (value if math.isfinite(value) else {NUMBER_KEY: repr(value)}), ()
    mapping: Final = _parsed(_DICT, value)
    if mapping is not None:
        if not all(isinstance(key, str) for key in mapping):
            return _opaque(value), ()
        converted: Final[dict[str, object]] = {str(key): None for key in mapping}
        return converted, [(partial(converted.__setitem__, str(key)), item) for key, item in mapping.items()]
    items: Final = _items(value)
    if items is not None:
        listed: Final[list[object]] = [None] * len(items)
        return listed, [(partial(listed.__setitem__, index), item) for index, item in enumerate(items)]
    return _opaque(value), ()


def _opaque(value: object) -> Mapping[str, object]:
    return {"$py": str(value), "$type": type(value).__name__}


def moment(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, datetime):
        return None
    aware: Final = value.utcoffset() is not None
    return {"us": (value - (_EPOCH_AWARE if aware else _EPOCH_NAIVE)) // _ONE_MICROSECOND, "aware": aware}


def wall_moment(value: datetime | None) -> Mapping[str, object] | None:
    return None if value is None else moment(value.replace(tzinfo=None))


def _settings() -> Mapping[str, object]:
    return {
        "emit_stream_label": litellm.prometheus_emit_stream_label is True,
        "custom_tags": list(litellm.custom_prometheus_tags or ()),
        "disable_end_user_cost_tracking": bool(litellm.disable_end_user_cost_tracking),
        "end_user_prometheus_only": litellm.enable_end_user_cost_tracking_prometheus_only is True,
        "end_user_max_series": litellm.prometheus_end_user_metrics_max_series_per_metric,
        "end_user_ttl_seconds": litellm.prometheus_end_user_metrics_ttl_seconds,
        "end_user_cleanup_interval_seconds": litellm.prometheus_end_user_metrics_cleanup_interval_seconds,
    }


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return {}


def residual(
    kwargs: Mapping[str, object], input_sequence_length_label: bool, now: datetime | None
) -> Mapping[str, object]:
    payload: Final = _mapping_or_empty(kwargs.get("standard_logging_object"))
    return {
        "settings": _settings(),
        "emit_input_sequence_length_label": input_sequence_length_label,
        "payload_labels": {key: json_safe(payload[key]) for key in _PAYLOAD_LABEL_KEYS if key in payload},
        "now": moment(now),
    }


def _labelnames(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    labelnames: Final = _items(kwargs.get("labelnames") or ())
    return () if labelnames is None else tuple(str(name) for name in labelnames)


def native_metric_factory(
    store: _native.NativePrometheusStore,
    kind: PrometheusKind,
    is_enabled: Callable[[str], bool],
    exclude_labels: frozenset[str],
    limits_enabled: bool,
    shares_overflow_series: bool,
    forwarding: ForwardingStore | None = None,
) -> Callable[..., _native.NativePrometheusMetric | NoOpMetric]:
    def factory(
        *args: object,
        **kwargs: object,  # kwargs-ok: mirrors the prometheus_client constructors it replaces
    ) -> _native.NativePrometheusMetric | NoOpMetric:
        name: Final = str(args[0] if args else kwargs.get("name", ""))
        if not is_enabled(name):
            return NoOpMetric()
        documentation: Final = str(args[1] if len(args) > 1 else kwargs.get("documentation", ""))
        original: Final = _labelnames(kwargs)
        kept: Final = tuple(label for label in original if label not in exclude_labels)
        wrapped: Final = bool(original) and (kept != original or limits_enabled)
        mode: Final = str(kwargs.get("multiprocess_mode", "all"))
        if mode not in prometheus_metrics.Gauge._MULTIPROC_MODES:  # pyright: ignore[reportPrivateUsage]  # the client's own mode table
            raise ValueError("Invalid multiprocess mode: " + mode)
        bounds: Final = _items(kwargs.get("buckets", prometheus_metrics.Histogram.DEFAULT_BUCKETS))
        buckets: Final = (
            [] if bounds is None else [float(bound) for bound in bounds]  # pyright: ignore[reportArgumentType]  # prometheus_client calls float() on each bound the same way
        )
        metric: Final = store.metric(
            kind,
            name,
            documentation,
            list(original),
            [label for label in original if label in exclude_labels],
            buckets if kind == "histogram" else [],
            mode in _MOST_RECENT_MODES,
            wrapped,
            shares_overflow_series,
        )
        if forwarding is None:
            REGISTRY.register(NativeFamilyCollector(metric))
        else:
            forwarding.add(
                name, forwarded_metric(kind, name, documentation, kept, buckets, mode), wrapped, shares_overflow_series
            )
        return metric

    return factory


class NativeFamilyCollector(Collector):
    def __init__(self, metric: _native.NativePrometheusMetric) -> None:
        self._metric: Final = metric

    def describe(self) -> Iterable[Metric]:
        name, documentation, kind, *_ = self._metric.collect()
        return [Metric(name, documentation, kind)]

    def collect(self) -> Iterable[Metric]:
        name, documentation, kind, labelnames, buckets, series = self._metric.collect()
        return [
            _family(
                name,
                documentation,
                kind,
                labelnames,
                buckets,
                series,
                prometheus_metrics._use_created,  # pyright: ignore[reportPrivateUsage]  # the client's own created-series switch
            )
        ]


def _family(
    name: str,
    documentation: str,
    kind: str,
    labelnames: Sequence[str],
    buckets: Sequence[float],
    series: Sequence[tuple[Sequence[str], float, float, Sequence[float], float]],
    created: bool,
) -> Metric:
    metric: Final = Metric(name, documentation, kind)
    for values, value, total, counts, created_at in series:
        labels = dict(zip(labelnames, values))
        if kind == "counter":
            metric.add_sample(name + "_total", labels, value)
        elif kind == "gauge":
            metric.add_sample(name, labels, value)
        else:
            cumulative = 0.0
            for bound, count in zip(buckets, counts):
                cumulative += count
                metric.add_sample(name + "_bucket", {**labels, "le": floatToGoString(bound)}, cumulative)
            metric.add_sample(name + "_count", labels, cumulative)
            if buckets[0] >= 0:
                metric.add_sample(name + "_sum", labels, total)
        if created and kind != "gauge":
            metric.add_sample(name + "_created", labels, created_at)
    return metric


class NativeStore:
    __slots__ = ("forwarding", "store")

    def __init__(self, store: _native.NativePrometheusStore, forwarding: ForwardingStore | None = None) -> None:
        self.store: Final = store
        self.forwarding: Final = forwarding

    @classmethod
    def open(cls, logger: PrometheusLogger, admissions: object) -> NativeStore:
        limits: Final = logger._series_limits  # pyright: ignore[reportPrivateUsage]  # the logger's resolved caps
        store: Final = _native.NativePrometheusStore(
            limits.max_series, limits.ttl_seconds, limits.cleanup_interval_seconds
        )
        if not isinstance(admissions, SharedPrometheusSeriesAdmissions):
            return cls(store)
        forwarding: Final = ForwardingStore(admissions, limits.max_series)
        store.forward(forwarding)
        return cls(store, forwarding)

    def flush(self) -> None:
        self.store.flush()

    def factory(
        self, metric_class: type[prometheus_metrics.MetricWrapperBase], logger: PrometheusLogger, shares: bool
    ) -> Callable[..., _native.NativePrometheusMetric | NoOpMetric]:
        return native_metric_factory(
            self.store,
            _KINDS[str(metric_class._type)],  # pyright: ignore[reportPrivateUsage]  # the client's own kind name
            logger._is_metric_enabled,  # pyright: ignore[reportPrivateUsage]  # the logger's enable rule
            logger.exclude_labels,
            logger._series_limits.enabled,  # pyright: ignore[reportPrivateUsage]  # the logger's resolved caps
            shares,
            self.forwarding,
        )

    def attach(self, logger: PrometheusLogger) -> None:
        for metric in _DEFINED_METRICS:
            if hasattr(PrometheusMetricLabels, metric):
                self.store.declare_labels(metric, list(logger.get_labels_for_metric(metric)))

    def op(self, name: NativeOp, facts: Mapping[str, object]) -> None:
        self.store.op(
            name, json.dumps({**{key: json_safe(value) for key, value in facts.items()}, "settings": _settings()})
        )

    def track_end_user(
        self, metric: str, values: Sequence[object], max_series: int | None, ttl_seconds: float | None
    ) -> None:
        self.store.track_end_user(
            metric,
            [None if value is None else str(value) for value in values],
            max_series,
            ttl_seconds,
            litellm.prometheus_end_user_metrics_cleanup_interval_seconds,
        )


_STOP_ERRORS: Final[Mapping[str, type[Exception]]] = {
    "value": ValueError,
    "key": KeyError,
    "type": TypeError,
    "runtime": RuntimeError,
}
_FIRST_STEP: Final[Mapping[NativeEvent, NativeStep]] = {"success": "success_head", "failure": "failure"}
_COMPANIONS: Final = count()


class NativeStop(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: NativeStep
    kind: Literal["value", "key", "type", "runtime", "named"]
    message: str
    error_type: str | None = None


def _named_error(error_type: str, text: str) -> Exception:
    found: Final[object] = getattr(builtins, error_type, None)
    base: Final = found if isinstance(found, type) and issubclass(found, Exception) else Exception

    def text_of(_error: BaseException) -> str:
        return text

    named: Final = type(error_type, (base,), {"__module__": "builtins", "__str__": text_of})
    return named(text)


@dataclass(frozen=True, slots=True)
class Stopped:
    step: NativeStep
    error: Exception


@dataclass(frozen=True, slots=True, eq=False)
class NativeTicket:
    companion: RustLogger
    call_id: object
    event: NativeEvent
    wire: Mapping[str, object]
    after: Callable[[], Awaitable[object]]
    budget: Callable[[], Awaitable[object]] | None
    view: object = None
    claims: Iterator[int] = field(default_factory=count)

    def claim(self, companion: RustLogger, kwargs: object) -> bool:
        return (
            companion is self.companion
            and _mapping_or_empty(kwargs).get("litellm_call_id") == self.call_id
            and next(self.claims) == 0
        )

    def stopped(self, error: RuntimeError) -> Stopped:
        try:
            stop: Final = NativeStop.model_validate_json(str(error))
        except ValidationError:
            return Stopped(step=_FIRST_STEP[self.event], error=error)
        if stop.kind == "named" and stop.error_type is not None:
            return Stopped(step=stop.step, error=_named_error(stop.error_type, stop.message))
        return Stopped(step=stop.step, error=_STOP_ERRORS.get(stop.kind, RuntimeError)(stop.message))


_TICKET: Final[ContextVar[NativeTicket | None]] = ContextVar("prometheus_native_ticket", default=None)


class _AttemptView:
    __slots__ = ("_view", "deprecated_kwargs")

    def __init__(self, view: object, wire: Mapping[str, object]) -> None:
        self._view: Final = view
        self.deprecated_kwargs: Final = wire

    def __getattr__(self, name: str) -> object:
        return getattr(self._view, name)


def disarm() -> None:
    _ = _TICKET.set(None)


def companion_in_line(logger: PrometheusLogger, line: Sequence[object]) -> bool:
    companion: Final = logger.v2_companion
    return companion is not None and any(item is companion for item in line)


def arm(
    companion: object,
    kwargs: Mapping[str, object],
    event: NativeEvent,
    after: Callable[[], Awaitable[object]],
    budget: Callable[[], Awaitable[object]] | None = None,
    input_sequence_length_label: bool = False,
    now: datetime | None = None,
) -> None:
    if not isinstance(companion, RustLogger):
        return
    _ = _TICKET.set(
        NativeTicket(
            companion=companion,
            call_id=kwargs.get("litellm_call_id"),
            event=event,
            wire=residual(kwargs, input_sequence_length_label, now),
            after=after,
            budget=budget,
        )
    )


def _no_legacy(_callback: object, _kwargs: object) -> None:
    return None


async def run_alone(
    companion: object,
    kwargs: Mapping[str, object],
    event: NativeEvent,
    response: object,
    start_time: object,
    end_time: object,
) -> None:
    from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, Logging

    if not isinstance(companion, RustLogger):
        return
    owner: Final = Logging(
        model=str(kwargs.get("model", "")),
        messages=[],
        stream=kwargs.get("stream") is True,
        call_type=str(kwargs.get("call_type") or "acompletion"),
        start_time=start_time if isinstance(start_time, datetime) else datetime.now(timezone.utc),
        litellm_call_id=str(kwargs.get("litellm_call_id", "")),
        function_id="prometheus",
    )
    owner.model_call_details = dict(kwargs)
    args: Final = HookArgs(
        request=kwargs,
        result=response,
        start_time=start_time,
        end_time=end_time,
        outcome=Outcome.success if event == "success" else Outcome.failure,
        logging_obj=owner,
    )
    _ = await run_subscribers(
        hook_subscribers(Hook.attempt_ended, (companion,), admit=Admit.every_entry),
        with_sources(KwargsFields(kwargs, LOGGING_KEYS), DERIVED_ATTEMPT_FIELDS, args),
        _no_legacy,
        call=owner.v2_call_facts(end_time),
    )


def replay_log(level: str, message: str, args: list[str]) -> None:
    (verbose_logger.debug if level == "debug" else verbose_logger.error)(message, *args)


def _count_callback_failure(owner: object) -> None:
    try:
        for callback in litellm.logging_callback_manager.get_all_callbacks():
            increment: object = getattr(callback, "increment_callback_logging_failure", None)
            if callable(increment):
                _ = increment(callback_name=type(owner).__name__)
                break
    except Exception as e:  # noqa: BLE001  # mirrors Logging._handle_callback_failure
        verbose_logger.debug("Error in _handle_callback_failure: %s", e)


def _report_success_failure(owner: object, error: Exception) -> None:
    try:
        raise error
    except Exception:  # noqa: BLE001  # logged with its traceback the way the legacy logging line does
        verbose_logger.error(
            "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging %s", traceback.format_exc()
        )
    _count_callback_failure(owner)


async def _settle_success(owner: object, ticket: NativeTicket, stopped: Stopped | None) -> None:
    step: Final = None if stopped is None else stopped.step
    try:
        if step != "success_head" and ticket.budget is not None:
            _ = await ticket.budget()
        if step in (None, "success_tail"):
            _ = await ticket.after()
    except Exception as error:  # noqa: BLE001  # the legacy logging line counts and logs it the same way
        _report_success_failure(owner, error)
        return
    if stopped is not None:
        _report_success_failure(owner, stopped.error)


async def _settle_failure(ticket: NativeTicket, stopped: Stopped | None) -> None:
    try:
        if stopped is not None:
            raise stopped.error
        _ = await ticket.after()
    except Exception as e:  # noqa: BLE001  # the legacy failure handler logs every error this way
        verbose_logger.exception("prometheus Layer Error(): Exception occured - %s", e)


class PrometheusCompanion(RustLogger):
    def __init__(
        self,
        native: _native._NativePrometheusCallback,  # pyright: ignore[reportPrivateUsage]  # the stub's callback protocol
        owner: PrometheusLogger,
        flush: Callable[[], None],
    ) -> None:
        super().__init__(native)
        self.owner: Final = weakref.ref(owner)
        self.serial: Final = next(_COMPANIONS)
        self.flush: Final = flush

    async def on_attempt_ended(self, view: object, ctx: CallContext[object]) -> None:
        ticket: Final = _TICKET.get()
        kwargs: Final = getattr(view, Field.deprecated_kwargs.value, None)
        if ticket is None or not ticket.claim(self, kwargs):
            return
        stopped: Final = await self._recorded(replace(ticket, view=_AttemptView(view, ticket.wire)), ctx)
        if ticket.event == "failure":
            await _settle_failure(ticket, stopped)
        else:
            await _settle_success(self.owner(), ticket, stopped)
        return

    async def _recorded(self, ticket: NativeTicket, ctx: CallContext[object]) -> Stopped | None:
        try:
            _ = await self.run_rust(Hook.attempt_ended, self.subscriptions[Hook.attempt_ended].reads, ticket.view, ctx)
        except RuntimeError as error:
            return ticket.stopped(error)
        finally:
            self.flush()
        return None


def companion(logger: PrometheusLogger, store: NativeStore) -> PrometheusCompanion:
    handle: Final = store.store.callback(replay_log)
    generated: Final = type(rust_logger(handle))
    built: Final = type(generated.__name__, (PrometheusCompanion, generated), {})
    return built(handle, logger, store.flush)
