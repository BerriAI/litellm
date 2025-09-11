import sentry_sdk
from sentry_sdk.crons import capture_checkin, MonitorStatus
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.integrations.celery.utils import (
    _get_humanized_interval,
    _now_seconds_since_epoch,
)
from sentry_sdk.utils import (
    logger,
    match_regex_list,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Optional, TypeVar, Union
    from sentry_sdk._types import (
        MonitorConfig,
        MonitorConfigScheduleType,
        MonitorConfigScheduleUnit,
    )

    F = TypeVar("F", bound=Callable[..., Any])


try:
    from celery import Task, Celery  # type: ignore
    from celery.beat import Scheduler  # type: ignore
    from celery.schedules import crontab, schedule  # type: ignore
    from celery.signals import (  # type: ignore
        task_failure,
        task_success,
        task_retry,
    )
except ImportError:
    raise DidNotEnable("Celery not installed")

try:
    from redbeat.schedulers import RedBeatScheduler  # type: ignore
except ImportError:
    RedBeatScheduler = None


def _get_headers(task):
    # type: (Task) -> dict[str, Any]
    headers = task.request.get("headers") or {}

    # flatten nested headers
    if "headers" in headers:
        headers.update(headers["headers"])
        del headers["headers"]

    headers.update(task.request.get("properties") or {})

    return headers


def _get_monitor_config(celery_schedule, app, monitor_name):
    # type: (Any, Celery, str) -> MonitorConfig
    monitor_config = {}  # type: MonitorConfig
    schedule_type = None  # type: Optional[MonitorConfigScheduleType]
    schedule_value = None  # type: Optional[Union[str, int]]
    schedule_unit = None  # type: Optional[MonitorConfigScheduleUnit]

    if isinstance(celery_schedule, crontab):
        schedule_type = "crontab"
        schedule_value = (
            "{0._orig_minute} "
            "{0._orig_hour} "
            "{0._orig_day_of_month} "
            "{0._orig_month_of_year} "
            "{0._orig_day_of_week}".format(celery_schedule)
        )
    elif isinstance(celery_schedule, schedule):
        schedule_type = "interval"
        (schedule_value, schedule_unit) = _get_humanized_interval(
            celery_schedule.seconds
        )

        if schedule_unit == "second":
            logger.warning(
                "Intervals shorter than one minute are not supported by Sentry Crons. Monitor '%s' has an interval of %s seconds. Use the `exclude_beat_tasks` option in the celery integration to exclude it.",
                monitor_name,
                schedule_value,
            )
            return {}

    else:
        logger.warning(
            "Celery schedule type '%s' not supported by Sentry Crons.",
            type(celery_schedule),
        )
        return {}

    monitor_config["schedule"] = {}
    monitor_config["schedule"]["type"] = schedule_type
    monitor_config["schedule"]["value"] = schedule_value

    if schedule_unit is not None:
        monitor_config["schedule"]["unit"] = schedule_unit

    monitor_config["timezone"] = (
        (
            hasattr(celery_schedule, "tz")
            and celery_schedule.tz is not None
            and str(celery_schedule.tz)
        )
        or app.timezone
        or "UTC"
    )

    return monitor_config


def _apply_crons_data_to_schedule_entry(scheduler, schedule_entry, integration):
    # type: (Any, Any, sentry_sdk.integrations.celery.CeleryIntegration) -> None
    """
    Add Sentry Crons information to the schedule_entry headers.
    """
    if not integration.monitor_beat_tasks:
        return

    monitor_name = schedule_entry.name

    task_should_be_excluded = match_regex_list(
        monitor_name, integration.exclude_beat_tasks
    )
    if task_should_be_excluded:
        return

    celery_schedule = schedule_entry.schedule
    app = scheduler.app

    monitor_config = _get_monitor_config(celery_schedule, app, monitor_name)

    is_supported_schedule = bool(monitor_config)
    if not is_supported_schedule:
        return

    headers = schedule_entry.options.pop("headers", {})
    headers.update(
        {
            "sentry-monitor-slug": monitor_name,
            "sentry-monitor-config": monitor_config,
        }
    )

    check_in_id = capture_checkin(
        monitor_slug=monitor_name,
        monitor_config=monitor_config,
        status=MonitorStatus.IN_PROGRESS,
    )
    headers.update({"sentry-monitor-check-in-id": check_in_id})

    # Set the Sentry configuration in the options of the ScheduleEntry.
    # Those will be picked up in `apply_async` and added to the headers.
    schedule_entry.options["headers"] = headers


def _wrap_beat_scheduler(original_function):
    # type: (Callable[..., Any]) -> Callable[..., Any]
    """
    Makes sure that:
    - a new Sentry trace is started for each task started by Celery Beat and
      it is propagated to the task.
    - the Sentry Crons information is set in the Celery Beat task's
      headers so that is is monitored with Sentry Crons.

    After the patched function is called,
    Celery Beat will call apply_async to put the task in the queue.
    """
    # Patch only once
    # Can't use __name__ here, because some of our tests mock original_apply_entry
    already_patched = "sentry_patched_scheduler" in str(original_function)
    if already_patched:
        return original_function

    from sentry_sdk.integrations.celery import CeleryIntegration

    def sentry_patched_scheduler(*args, **kwargs):
        # type: (*Any, **Any) -> None
        integration = sentry_sdk.get_client().get_integration(CeleryIntegration)
        if integration is None:
            return original_function(*args, **kwargs)

        # Tasks started by Celery Beat start a new Trace
        scope = sentry_sdk.get_isolation_scope()
        scope.set_new_propagation_context()
        scope._name = "celery-beat"

        scheduler, schedule_entry = args
        _apply_crons_data_to_schedule_entry(scheduler, schedule_entry, integration)

        return original_function(*args, **kwargs)

    return sentry_patched_scheduler


def _patch_beat_apply_entry():
    # type: () -> None
    Scheduler.apply_entry = _wrap_beat_scheduler(Scheduler.apply_entry)


def _patch_redbeat_maybe_due():
    # type: () -> None
    if RedBeatScheduler is None:
        return

    RedBeatScheduler.maybe_due = _wrap_beat_scheduler(RedBeatScheduler.maybe_due)


def _setup_celery_beat_signals(monitor_beat_tasks):
    # type: (bool) -> None
    if monitor_beat_tasks:
        task_success.connect(crons_task_success)
        task_failure.connect(crons_task_failure)
        task_retry.connect(crons_task_retry)


def crons_task_success(sender, **kwargs):
    # type: (Task, dict[Any, Any]) -> None
    logger.debug("celery_task_success %s", sender)
    headers = _get_headers(sender)

    if "sentry-monitor-slug" not in headers:
        return

    monitor_config = headers.get("sentry-monitor-config", {})

    start_timestamp_s = headers.get("sentry-monitor-start-timestamp-s")

    capture_checkin(
        monitor_slug=headers["sentry-monitor-slug"],
        monitor_config=monitor_config,
        check_in_id=headers["sentry-monitor-check-in-id"],
        duration=(
            _now_seconds_since_epoch() - float(start_timestamp_s)
            if start_timestamp_s
            else None
        ),
        status=MonitorStatus.OK,
    )


def crons_task_failure(sender, **kwargs):
    # type: (Task, dict[Any, Any]) -> None
    logger.debug("celery_task_failure %s", sender)
    headers = _get_headers(sender)

    if "sentry-monitor-slug" not in headers:
        return

    monitor_config = headers.get("sentry-monitor-config", {})

    start_timestamp_s = headers.get("sentry-monitor-start-timestamp-s")

    capture_checkin(
        monitor_slug=headers["sentry-monitor-slug"],
        monitor_config=monitor_config,
        check_in_id=headers["sentry-monitor-check-in-id"],
        duration=(
            _now_seconds_since_epoch() - float(start_timestamp_s)
            if start_timestamp_s
            else None
        ),
        status=MonitorStatus.ERROR,
    )


def crons_task_retry(sender, **kwargs):
    # type: (Task, dict[Any, Any]) -> None
    logger.debug("celery_task_retry %s", sender)
    headers = _get_headers(sender)

    if "sentry-monitor-slug" not in headers:
        return

    monitor_config = headers.get("sentry-monitor-config", {})

    start_timestamp_s = headers.get("sentry-monitor-start-timestamp-s")

    capture_checkin(
        monitor_slug=headers["sentry-monitor-slug"],
        monitor_config=monitor_config,
        check_in_id=headers["sentry-monitor-check-in-id"],
        duration=(
            _now_seconds_since_epoch() - float(start_timestamp_s)
            if start_timestamp_s
            else None
        ),
        status=MonitorStatus.ERROR,
    )
