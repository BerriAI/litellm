import sys

import sentry_sdk
from sentry_sdk.consts import OP, SPANSTATUS
from sentry_sdk.integrations import _check_minimum_version, DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    SENSITIVE_DATA_SUBSTITUTE,
    parse_version,
    reraise,
)

try:
    import arq.worker
    from arq.version import VERSION as ARQ_VERSION
    from arq.connections import ArqRedis
    from arq.worker import JobExecutionFailed, Retry, RetryJob, Worker
except ImportError:
    raise DidNotEnable("Arq is not installed")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Union

    from sentry_sdk._types import EventProcessor, Event, ExcInfo, Hint

    from arq.cron import CronJob
    from arq.jobs import Job
    from arq.typing import WorkerCoroutine
    from arq.worker import Function

ARQ_CONTROL_FLOW_EXCEPTIONS = (JobExecutionFailed, Retry, RetryJob)


class ArqIntegration(Integration):
    identifier = "arq"
    origin = f"auto.queue.{identifier}"

    @staticmethod
    def setup_once():
        # type: () -> None

        try:
            if isinstance(ARQ_VERSION, str):
                version = parse_version(ARQ_VERSION)
            else:
                version = ARQ_VERSION.version[:2]

        except (TypeError, ValueError):
            version = None

        _check_minimum_version(ArqIntegration, version)

        patch_enqueue_job()
        patch_run_job()
        patch_create_worker()

        ignore_logger("arq.worker")


def patch_enqueue_job():
    # type: () -> None
    old_enqueue_job = ArqRedis.enqueue_job
    original_kwdefaults = old_enqueue_job.__kwdefaults__

    async def _sentry_enqueue_job(self, function, *args, **kwargs):
        # type: (ArqRedis, str, *Any, **Any) -> Optional[Job]
        integration = sentry_sdk.get_client().get_integration(ArqIntegration)
        if integration is None:
            return await old_enqueue_job(self, function, *args, **kwargs)

        with sentry_sdk.start_span(
            op=OP.QUEUE_SUBMIT_ARQ, name=function, origin=ArqIntegration.origin
        ):
            return await old_enqueue_job(self, function, *args, **kwargs)

    _sentry_enqueue_job.__kwdefaults__ = original_kwdefaults
    ArqRedis.enqueue_job = _sentry_enqueue_job


def patch_run_job():
    # type: () -> None
    old_run_job = Worker.run_job

    async def _sentry_run_job(self, job_id, score):
        # type: (Worker, str, int) -> None
        integration = sentry_sdk.get_client().get_integration(ArqIntegration)
        if integration is None:
            return await old_run_job(self, job_id, score)

        with sentry_sdk.isolation_scope() as scope:
            scope._name = "arq"
            scope.clear_breadcrumbs()

            transaction = Transaction(
                name="unknown arq task",
                status="ok",
                op=OP.QUEUE_TASK_ARQ,
                source=TRANSACTION_SOURCE_TASK,
                origin=ArqIntegration.origin,
            )

            with sentry_sdk.start_transaction(transaction):
                return await old_run_job(self, job_id, score)

    Worker.run_job = _sentry_run_job


def _capture_exception(exc_info):
    # type: (ExcInfo) -> None
    scope = sentry_sdk.get_current_scope()

    if scope.transaction is not None:
        if exc_info[0] in ARQ_CONTROL_FLOW_EXCEPTIONS:
            scope.transaction.set_status(SPANSTATUS.ABORTED)
            return

        scope.transaction.set_status(SPANSTATUS.INTERNAL_ERROR)

    event, hint = event_from_exception(
        exc_info,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": ArqIntegration.identifier, "handled": False},
    )
    sentry_sdk.capture_event(event, hint=hint)


def _make_event_processor(ctx, *args, **kwargs):
    # type: (Dict[Any, Any], *Any, **Any) -> EventProcessor
    def event_processor(event, hint):
        # type: (Event, Hint) -> Optional[Event]

        with capture_internal_exceptions():
            scope = sentry_sdk.get_current_scope()
            if scope.transaction is not None:
                scope.transaction.name = ctx["job_name"]
                event["transaction"] = ctx["job_name"]

            tags = event.setdefault("tags", {})
            tags["arq_task_id"] = ctx["job_id"]
            tags["arq_task_retry"] = ctx["job_try"] > 1
            extra = event.setdefault("extra", {})
            extra["arq-job"] = {
                "task": ctx["job_name"],
                "args": (
                    args if should_send_default_pii() else SENSITIVE_DATA_SUBSTITUTE
                ),
                "kwargs": (
                    kwargs if should_send_default_pii() else SENSITIVE_DATA_SUBSTITUTE
                ),
                "retry": ctx["job_try"],
            }

        return event

    return event_processor


def _wrap_coroutine(name, coroutine):
    # type: (str, WorkerCoroutine) -> WorkerCoroutine

    async def _sentry_coroutine(ctx, *args, **kwargs):
        # type: (Dict[Any, Any], *Any, **Any) -> Any
        integration = sentry_sdk.get_client().get_integration(ArqIntegration)
        if integration is None:
            return await coroutine(ctx, *args, **kwargs)

        sentry_sdk.get_isolation_scope().add_event_processor(
            _make_event_processor({**ctx, "job_name": name}, *args, **kwargs)
        )

        try:
            result = await coroutine(ctx, *args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            _capture_exception(exc_info)
            reraise(*exc_info)

        return result

    return _sentry_coroutine


def patch_create_worker():
    # type: () -> None
    old_create_worker = arq.worker.create_worker

    @ensure_integration_enabled(ArqIntegration, old_create_worker)
    def _sentry_create_worker(*args, **kwargs):
        # type: (*Any, **Any) -> Worker
        settings_cls = args[0]

        if isinstance(settings_cls, dict):
            if "functions" in settings_cls:
                settings_cls["functions"] = [
                    _get_arq_function(func) for func in settings_cls["functions"]
                ]
            if "cron_jobs" in settings_cls:
                settings_cls["cron_jobs"] = [
                    _get_arq_cron_job(cron_job)
                    for cron_job in settings_cls["cron_jobs"]
                ]

        if hasattr(settings_cls, "functions"):
            settings_cls.functions = [
                _get_arq_function(func) for func in settings_cls.functions
            ]
        if hasattr(settings_cls, "cron_jobs"):
            settings_cls.cron_jobs = [
                _get_arq_cron_job(cron_job) for cron_job in settings_cls.cron_jobs
            ]

        if "functions" in kwargs:
            kwargs["functions"] = [
                _get_arq_function(func) for func in kwargs["functions"]
            ]
        if "cron_jobs" in kwargs:
            kwargs["cron_jobs"] = [
                _get_arq_cron_job(cron_job) for cron_job in kwargs["cron_jobs"]
            ]

        return old_create_worker(*args, **kwargs)

    arq.worker.create_worker = _sentry_create_worker


def _get_arq_function(func):
    # type: (Union[str, Function, WorkerCoroutine]) -> Function
    arq_func = arq.worker.func(func)
    arq_func.coroutine = _wrap_coroutine(arq_func.name, arq_func.coroutine)

    return arq_func


def _get_arq_cron_job(cron_job):
    # type: (CronJob) -> CronJob
    cron_job.coroutine = _wrap_coroutine(cron_job.name, cron_job.coroutine)

    return cron_job
