import uuid

import sentry_sdk

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from sentry_sdk._types import Event, MonitorConfig


def _create_check_in_event(
    monitor_slug=None,  # type: Optional[str]
    check_in_id=None,  # type: Optional[str]
    status=None,  # type: Optional[str]
    duration_s=None,  # type: Optional[float]
    monitor_config=None,  # type: Optional[MonitorConfig]
):
    # type: (...) -> Event
    options = sentry_sdk.get_client().options
    check_in_id = check_in_id or uuid.uuid4().hex  # type: str

    check_in = {
        "type": "check_in",
        "monitor_slug": monitor_slug,
        "check_in_id": check_in_id,
        "status": status,
        "duration": duration_s,
        "environment": options.get("environment", None),
        "release": options.get("release", None),
    }  # type: Event

    if monitor_config:
        check_in["monitor_config"] = monitor_config

    return check_in


def capture_checkin(
    monitor_slug=None,  # type: Optional[str]
    check_in_id=None,  # type: Optional[str]
    status=None,  # type: Optional[str]
    duration=None,  # type: Optional[float]
    monitor_config=None,  # type: Optional[MonitorConfig]
):
    # type: (...) -> str
    check_in_event = _create_check_in_event(
        monitor_slug=monitor_slug,
        check_in_id=check_in_id,
        status=status,
        duration_s=duration,
        monitor_config=monitor_config,
    )

    sentry_sdk.capture_event(check_in_event)

    return check_in_event["check_in_id"]
