import sys
import warnings

from collections.abc import Mapping
from functools import wraps

import sentry_sdk
from sentry_sdk import isolation_scope
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP, SPANSTATUS, SPANDATA
from sentry_sdk.integrations import _check_minimum_version, Integration, DidNotEnable
from sentry_sdk.integrations.celery.beat import (
    _patch_beat_apply_entry,
    _patch_redbeat_maybe_due,
    _setup_celery_beat_signals,
)
from sentry_sdk.integrations.celery.utils import _now_seconds_since_epoch
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import BAGGAGE_HEADER_NAME, TRANSACTION_SOURCE_TASK
from sentry_sdk.tracing_utils import Baggage
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    reraise,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import List
    from typing import Optional
    from typing import TypeVar
    from typing import Union

    from sentry_sdk._types import EventProcessor, Event, Hint, ExcInfo
    from sentry_sdk.tracing import Span

    F = TypeVar("F", bound=Callable[..., Any])


try:
    from celery import VERSION as CELERY_VERSION  # type: ignore
    from celery.app.task import Task  # type: ignore
    from celery.app.trace import task_has_custom
    from celery.exceptions import (  # type: ignore
        Ignore,
        Reject,
        Retry,
        SoftTimeLimitExceeded,
    )
    from kombu import Producer  # type: ignore
except ImportError:
    raise DidNotEnable("Celery not installed")


CELERY_CONTROL_FLOW_EXCEPTIONS = (Retry, Ignore, Reject)


class CeleryIntegration(Integration):
    identifier = "celery"
    origin = f"auto.queue.{identifier}"

    def __init__(
        self,
        propagate_traces=True,
        monitor_beat_tasks=False,
        exclude_beat_tasks=None,
    ):
        # type: (bool, bool, Optional[List[str]]) -> None
        warnings.warn(
            "The `propagate_traces` parameter is deprecated. Please use `trace_propagation_targets` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.propagate_traces = propagate_traces
        self.monitor_beat_tasks = monitor_beat_tasks
        self.exclude_beat_tasks = exclude_beat_tasks

        _patch_beat_apply_entry()
        _patch_redbeat_maybe_due()
        _setup_celery_beat_signals(monitor_beat_tasks)

    @staticmethod
    def setup_once():
        # type: () -> None
        _check_minimum_version(CeleryIntegration, CELERY_VERSION)

        _patch_build_tracer()
        _patch_task_apply_async()
        _patch_celery_send_task()
        _patch_worker_exit()
        _patch_producer_publish()

        # This logger logs every status of every task that ran on the worker.
        # Meaning that every task's breadcrumbs are full of stuff like "Task
        # <foo> raised unexpected <bar>".
        ignore_logger("celery.worker.job")
        ignore_logger("celery.app.trace")

        # This is stdout/err redirected to a logger, can't deal with this
        # (need event_level=logging.WARN to reproduce)
        ignore_logger("celery.redirected")


def _set_status(status):
    # type: (str) -> None
    with capture_internal_exceptions():
        scope = sentry_sdk.get_current_scope()
        if scope.span is not None:
            scope.span.set_status(status)


def _capture_exception(task, exc_info):
    # type: (Any, ExcInfo) -> None
    client = sentry_sdk.get_client()
    if client.get_integration(CeleryIntegration) is None:
        return

    if isinstance(exc_info[1], CELERY_CONTROL_FLOW_EXCEPTIONS):
        # ??? Doesn't map to anything
        _set_status("aborted")
        return

    _set_status("internal_error")

    if hasattr(task, "throws") and isinstance(exc_info[1], task.throws):
        return

    event, hint = event_from_exception(
        exc_info,
        client_options=client.options,
        mechanism={"type": "celery", "handled": False},
    )

    sentry_sdk.capture_event(event, hint=hint)


def _make_event_processor(task, uuid, args, kwargs, request=None):
    # type: (Any, Any, Any, Any, Optional[Any]) -> EventProcessor
    def event_processor(event, hint):
        # type: (Event, Hint) -> Optional[Event]

        with capture_internal_exceptions():
            tags = event.setdefault("tags", {})
            tags["celery_task_id"] = uuid
            extra = event.setdefault("extra", {})
            extra["celery-job"] = {
                "task_name": task.name,
                "args": args,
                "kwargs": kwargs,
            }

        if "exc_info" in hint:
            with capture_internal_exceptions():
                if issubclass(hint["exc_info"][0], SoftTimeLimitExceeded):
                    event["fingerprint"] = [
                        "celery",
                        "SoftTimeLimitExceeded",
                        getattr(task, "name", task),
                    ]

        return event

    return event_processor


def _update_celery_task_headers(original_headers, span, monitor_beat_tasks):
    # type: (dict[str, Any], Optional[Span], bool) -> dict[str, Any]
    """
    Updates the headers of the Celery task with the tracing information
    and eventually Sentry Crons monitoring information for beat tasks.
    """
    updated_headers = original_headers.copy()
    with capture_internal_exceptions():
        # if span is None (when the task was started by Celery Beat)
        # this will return the trace headers from the scope.
        headers = dict(
            sentry_sdk.get_isolation_scope().iter_trace_propagation_headers(span=span)
        )

        if monitor_beat_tasks:
            headers.update(
                {
                    "sentry-monitor-start-timestamp-s": "%.9f"
                    % _now_seconds_since_epoch(),
                }
            )

        # Add the time the task was enqueued to the headers
        # This is used in the consumer to calculate the latency
        updated_headers.update(
            {"sentry-task-enqueued-time": _now_seconds_since_epoch()}
        )

        if headers:
            existing_baggage = updated_headers.get(BAGGAGE_HEADER_NAME)
            sentry_baggage = headers.get(BAGGAGE_HEADER_NAME)

            combined_baggage = sentry_baggage or existing_baggage
            if sentry_baggage and existing_baggage:
                # Merge incoming and sentry baggage, where the sentry trace information
                # in the incoming baggage takes precedence and the third-party items
                # are concatenated.
                incoming = Baggage.from_incoming_header(existing_baggage)
                combined = Baggage.from_incoming_header(sentry_baggage)
                combined.sentry_items.update(incoming.sentry_items)
                combined.third_party_items = ",".join(
                    [
                        x
                        for x in [
                            combined.third_party_items,
                            incoming.third_party_items,
                        ]
                        if x is not None and x != ""
                    ]
                )
                combined_baggage = combined.serialize(include_third_party=True)

            updated_headers.update(headers)
            if combined_baggage:
                updated_headers[BAGGAGE_HEADER_NAME] = combined_baggage

            # https://github.com/celery/celery/issues/4875
            #
            # Need to setdefault the inner headers too since other
            # tracing tools (dd-trace-py) also employ this exact
            # workaround and we don't want to break them.
            updated_headers.setdefault("headers", {}).update(headers)
            if combined_baggage:
                updated_headers["headers"][BAGGAGE_HEADER_NAME] = combined_baggage

            # Add the Sentry options potentially added in `sentry_apply_entry`
            # to the headers (done when auto-instrumenting Celery Beat tasks)
            for key, value in updated_headers.items():
                if key.startswith("sentry-"):
                    updated_headers["headers"][key] = value

    return updated_headers


class NoOpMgr:
    def __enter__(self):
        # type: () -> None
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        # type: (Any, Any, Any) -> None
        return None


def _wrap_task_run(f):
    # type: (F) -> F
    @wraps(f)
    def apply_async(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        # Note: kwargs can contain headers=None, so no setdefault!
        # Unsure which backend though.
        integration = sentry_sdk.get_client().get_integration(CeleryIntegration)
        if integration is None:
            return f(*args, **kwargs)

        kwarg_headers = kwargs.get("headers") or {}
        propagate_traces = kwarg_headers.pop(
            "sentry-propagate-traces", integration.propagate_traces
        )

        if not propagate_traces:
            return f(*args, **kwargs)

        if isinstance(args[0], Task):
            task_name = args[0].name  # type: str
        elif len(args) > 1 and isinstance(args[1], str):
            task_name = args[1]
        else:
            task_name = "<unknown Celery task>"

        task_started_from_beat = sentry_sdk.get_isolation_scope()._name == "celery-beat"

        span_mgr = (
            sentry_sdk.start_span(
                op=OP.QUEUE_SUBMIT_CELERY,
                name=task_name,
                origin=CeleryIntegration.origin,
            )
            if not task_started_from_beat
            else NoOpMgr()
        )  # type: Union[Span, NoOpMgr]

        with span_mgr as span:
            kwargs["headers"] = _update_celery_task_headers(
                kwarg_headers, span, integration.monitor_beat_tasks
            )
            return f(*args, **kwargs)

    return apply_async  # type: ignore


def _wrap_tracer(task, f):
    # type: (Any, F) -> F

    # Need to wrap tracer for pushing the scope before prerun is sent, and
    # popping it after postrun is sent.
    #
    # This is the reason we don't use signals for hooking in the first place.
    # Also because in Celery 3, signal dispatch returns early if one handler
    # crashes.
    @wraps(f)
    @ensure_integration_enabled(CeleryIntegration, f)
    def _inner(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        with isolation_scope() as scope:
            scope._name = "celery"
            scope.clear_breadcrumbs()
            scope.add_event_processor(_make_event_processor(task, *args, **kwargs))

            transaction = None

            # Celery task objects are not a thing to be trusted. Even
            # something such as attribute access can fail.
            with capture_internal_exceptions():
                headers = args[3].get("headers") or {}
                transaction = continue_trace(
                    headers,
                    op=OP.QUEUE_TASK_CELERY,
                    name="unknown celery task",
                    source=TRANSACTION_SOURCE_TASK,
                    origin=CeleryIntegration.origin,
                )
                transaction.name = task.name
                transaction.set_status(SPANSTATUS.OK)

            if transaction is None:
                return f(*args, **kwargs)

            with sentry_sdk.start_transaction(
                transaction,
                custom_sampling_context={
                    "celery_job": {
                        "task": task.name,
                        # for some reason, args[1] is a list if non-empty but a
                        # tuple if empty
                        "args": list(args[1]),
                        "kwargs": args[2],
                    }
                },
            ):
                return f(*args, **kwargs)

    return _inner  # type: ignore


def _set_messaging_destination_name(task, span):
    # type: (Any, Span) -> None
    """Set "messaging.destination.name" tag for span"""
    with capture_internal_exceptions():
        delivery_info = task.request.delivery_info
        if delivery_info:
            routing_key = delivery_info.get("routing_key")
            if delivery_info.get("exchange") == "" and routing_key is not None:
                # Empty exchange indicates the default exchange, meaning the tasks
                # are sent to the queue with the same name as the routing key.
                span.set_data(SPANDATA.MESSAGING_DESTINATION_NAME, routing_key)


def _wrap_task_call(task, f):
    # type: (Any, F) -> F

    # Need to wrap task call because the exception is caught before we get to
    # see it. Also celery's reported stacktrace is untrustworthy.

    # functools.wraps is important here because celery-once looks at this
    # method's name. @ensure_integration_enabled internally calls functools.wraps,
    # but if we ever remove the @ensure_integration_enabled decorator, we need
    # to add @functools.wraps(f) here.
    # https://github.com/getsentry/sentry-python/issues/421
    @ensure_integration_enabled(CeleryIntegration, f)
    def _inner(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        try:
            with sentry_sdk.start_span(
                op=OP.QUEUE_PROCESS,
                name=task.name,
                origin=CeleryIntegration.origin,
            ) as span:
                _set_messaging_destination_name(task, span)

                latency = None
                with capture_internal_exceptions():
                    if (
                        task.request.headers is not None
                        and "sentry-task-enqueued-time" in task.request.headers
                    ):
                        latency = _now_seconds_since_epoch() - task.request.headers.pop(
                            "sentry-task-enqueued-time"
                        )

                if latency is not None:
                    span.set_data(SPANDATA.MESSAGING_MESSAGE_RECEIVE_LATENCY, latency)

                with capture_internal_exceptions():
                    span.set_data(SPANDATA.MESSAGING_MESSAGE_ID, task.request.id)

                with capture_internal_exceptions():
                    span.set_data(
                        SPANDATA.MESSAGING_MESSAGE_RETRY_COUNT, task.request.retries
                    )

                with capture_internal_exceptions():
                    span.set_data(
                        SPANDATA.MESSAGING_SYSTEM,
                        task.app.connection().transport.driver_type,
                    )

                return f(*args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            with capture_internal_exceptions():
                _capture_exception(task, exc_info)
            reraise(*exc_info)

    return _inner  # type: ignore


def _patch_build_tracer():
    # type: () -> None
    import celery.app.trace as trace  # type: ignore

    original_build_tracer = trace.build_tracer

    def sentry_build_tracer(name, task, *args, **kwargs):
        # type: (Any, Any, *Any, **Any) -> Any
        if not getattr(task, "_sentry_is_patched", False):
            # determine whether Celery will use __call__ or run and patch
            # accordingly
            if task_has_custom(task, "__call__"):
                type(task).__call__ = _wrap_task_call(task, type(task).__call__)
            else:
                task.run = _wrap_task_call(task, task.run)

            # `build_tracer` is apparently called for every task
            # invocation. Can't wrap every celery task for every invocation
            # or we will get infinitely nested wrapper functions.
            task._sentry_is_patched = True

        return _wrap_tracer(task, original_build_tracer(name, task, *args, **kwargs))

    trace.build_tracer = sentry_build_tracer


def _patch_task_apply_async():
    # type: () -> None
    Task.apply_async = _wrap_task_run(Task.apply_async)


def _patch_celery_send_task():
    # type: () -> None
    from celery import Celery

    Celery.send_task = _wrap_task_run(Celery.send_task)


def _patch_worker_exit():
    # type: () -> None

    # Need to flush queue before worker shutdown because a crashing worker will
    # call os._exit
    from billiard.pool import Worker  # type: ignore

    original_workloop = Worker.workloop

    def sentry_workloop(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        try:
            return original_workloop(*args, **kwargs)
        finally:
            with capture_internal_exceptions():
                if (
                    sentry_sdk.get_client().get_integration(CeleryIntegration)
                    is not None
                ):
                    sentry_sdk.flush()

    Worker.workloop = sentry_workloop


def _patch_producer_publish():
    # type: () -> None
    original_publish = Producer.publish

    @ensure_integration_enabled(CeleryIntegration, original_publish)
    def sentry_publish(self, *args, **kwargs):
        # type: (Producer, *Any, **Any) -> Any
        kwargs_headers = kwargs.get("headers", {})
        if not isinstance(kwargs_headers, Mapping):
            # Ensure kwargs_headers is a Mapping, so we can safely call get().
            # We don't expect this to happen, but it's better to be safe. Even
            # if it does happen, only our instrumentation breaks. This line
            # does not overwrite kwargs["headers"], so the original publish
            # method will still work.
            kwargs_headers = {}

        task_name = kwargs_headers.get("task")
        task_id = kwargs_headers.get("id")
        retries = kwargs_headers.get("retries")

        routing_key = kwargs.get("routing_key")
        exchange = kwargs.get("exchange")

        with sentry_sdk.start_span(
            op=OP.QUEUE_PUBLISH,
            name=task_name,
            origin=CeleryIntegration.origin,
        ) as span:
            if task_id is not None:
                span.set_data(SPANDATA.MESSAGING_MESSAGE_ID, task_id)

            if exchange == "" and routing_key is not None:
                # Empty exchange indicates the default exchange, meaning messages are
                # routed to the queue with the same name as the routing key.
                span.set_data(SPANDATA.MESSAGING_DESTINATION_NAME, routing_key)

            if retries is not None:
                span.set_data(SPANDATA.MESSAGING_MESSAGE_RETRY_COUNT, retries)

            with capture_internal_exceptions():
                span.set_data(
                    SPANDATA.MESSAGING_SYSTEM, self.connection.transport.driver_type
                )

            return original_publish(self, *args, **kwargs)

    Producer.publish = sentry_publish
