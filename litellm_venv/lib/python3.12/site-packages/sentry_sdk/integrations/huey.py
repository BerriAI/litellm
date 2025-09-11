import sys
from datetime import datetime

import sentry_sdk
from sentry_sdk.api import continue_trace, get_baggage, get_traceparent
from sentry_sdk.consts import OP, SPANSTATUS
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import (
    BAGGAGE_HEADER_NAME,
    SENTRY_TRACE_HEADER_NAME,
    TRANSACTION_SOURCE_TASK,
)
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    SENSITIVE_DATA_SUBSTITUTE,
    reraise,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union, TypeVar

    from sentry_sdk._types import EventProcessor, Event, Hint
    from sentry_sdk.utils import ExcInfo

    F = TypeVar("F", bound=Callable[..., Any])

try:
    from huey.api import Huey, Result, ResultGroup, Task, PeriodicTask
    from huey.exceptions import CancelExecution, RetryTask, TaskLockedException
except ImportError:
    raise DidNotEnable("Huey is not installed")


HUEY_CONTROL_FLOW_EXCEPTIONS = (CancelExecution, RetryTask, TaskLockedException)


class HueyIntegration(Integration):
    identifier = "huey"
    origin = f"auto.queue.{identifier}"

    @staticmethod
    def setup_once():
        # type: () -> None
        patch_enqueue()
        patch_execute()


def patch_enqueue():
    # type: () -> None
    old_enqueue = Huey.enqueue

    @ensure_integration_enabled(HueyIntegration, old_enqueue)
    def _sentry_enqueue(self, task):
        # type: (Huey, Task) -> Optional[Union[Result, ResultGroup]]
        with sentry_sdk.start_span(
            op=OP.QUEUE_SUBMIT_HUEY,
            name=task.name,
            origin=HueyIntegration.origin,
        ):
            if not isinstance(task, PeriodicTask):
                # Attach trace propagation data to task kwargs. We do
                # not do this for periodic tasks, as these don't
                # really have an originating transaction.
                task.kwargs["sentry_headers"] = {
                    BAGGAGE_HEADER_NAME: get_baggage(),
                    SENTRY_TRACE_HEADER_NAME: get_traceparent(),
                }
            return old_enqueue(self, task)

    Huey.enqueue = _sentry_enqueue


def _make_event_processor(task):
    # type: (Any) -> EventProcessor
    def event_processor(event, hint):
        # type: (Event, Hint) -> Optional[Event]

        with capture_internal_exceptions():
            tags = event.setdefault("tags", {})
            tags["huey_task_id"] = task.id
            tags["huey_task_retry"] = task.default_retries > task.retries
            extra = event.setdefault("extra", {})
            extra["huey-job"] = {
                "task": task.name,
                "args": (
                    task.args
                    if should_send_default_pii()
                    else SENSITIVE_DATA_SUBSTITUTE
                ),
                "kwargs": (
                    task.kwargs
                    if should_send_default_pii()
                    else SENSITIVE_DATA_SUBSTITUTE
                ),
                "retry": (task.default_retries or 0) - task.retries,
            }

        return event

    return event_processor


def _capture_exception(exc_info):
    # type: (ExcInfo) -> None
    scope = sentry_sdk.get_current_scope()

    if exc_info[0] in HUEY_CONTROL_FLOW_EXCEPTIONS:
        scope.transaction.set_status(SPANSTATUS.ABORTED)
        return

    scope.transaction.set_status(SPANSTATUS.INTERNAL_ERROR)
    event, hint = event_from_exception(
        exc_info,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": HueyIntegration.identifier, "handled": False},
    )
    scope.capture_event(event, hint=hint)


def _wrap_task_execute(func):
    # type: (F) -> F

    @ensure_integration_enabled(HueyIntegration, func)
    def _sentry_execute(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        try:
            result = func(*args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            _capture_exception(exc_info)
            reraise(*exc_info)

        return result

    return _sentry_execute  # type: ignore


def patch_execute():
    # type: () -> None
    old_execute = Huey._execute

    @ensure_integration_enabled(HueyIntegration, old_execute)
    def _sentry_execute(self, task, timestamp=None):
        # type: (Huey, Task, Optional[datetime]) -> Any
        with sentry_sdk.isolation_scope() as scope:
            with capture_internal_exceptions():
                scope._name = "huey"
                scope.clear_breadcrumbs()
                scope.add_event_processor(_make_event_processor(task))

            sentry_headers = task.kwargs.pop("sentry_headers", None)

            transaction = continue_trace(
                sentry_headers or {},
                name=task.name,
                op=OP.QUEUE_TASK_HUEY,
                source=TRANSACTION_SOURCE_TASK,
                origin=HueyIntegration.origin,
            )
            transaction.set_status(SPANSTATUS.OK)

            if not getattr(task, "_sentry_is_patched", False):
                task.execute = _wrap_task_execute(task.execute)
                task._sentry_is_patched = True

            with sentry_sdk.start_transaction(transaction):
                return old_execute(self, task, timestamp)

    Huey._execute = _sentry_execute
