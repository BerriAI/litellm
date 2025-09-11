import uuid
from datetime import datetime, timezone

from sentry_sdk.utils import format_timestamp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from typing import Union
    from typing import Any
    from typing import Dict

    from sentry_sdk._types import SessionStatus


def _minute_trunc(ts):
    # type: (datetime) -> datetime
    return ts.replace(second=0, microsecond=0)


def _make_uuid(
    val,  # type: Union[str, uuid.UUID]
):
    # type: (...) -> uuid.UUID
    if isinstance(val, uuid.UUID):
        return val
    return uuid.UUID(val)


class Session:
    def __init__(
        self,
        sid=None,  # type: Optional[Union[str, uuid.UUID]]
        did=None,  # type: Optional[str]
        timestamp=None,  # type: Optional[datetime]
        started=None,  # type: Optional[datetime]
        duration=None,  # type: Optional[float]
        status=None,  # type: Optional[SessionStatus]
        release=None,  # type: Optional[str]
        environment=None,  # type: Optional[str]
        user_agent=None,  # type: Optional[str]
        ip_address=None,  # type: Optional[str]
        errors=None,  # type: Optional[int]
        user=None,  # type: Optional[Any]
        session_mode="application",  # type: str
    ):
        # type: (...) -> None
        if sid is None:
            sid = uuid.uuid4()
        if started is None:
            started = datetime.now(timezone.utc)
        if status is None:
            status = "ok"
        self.status = status
        self.did = None  # type: Optional[str]
        self.started = started
        self.release = None  # type: Optional[str]
        self.environment = None  # type: Optional[str]
        self.duration = None  # type: Optional[float]
        self.user_agent = None  # type: Optional[str]
        self.ip_address = None  # type: Optional[str]
        self.session_mode = session_mode  # type: str
        self.errors = 0

        self.update(
            sid=sid,
            did=did,
            timestamp=timestamp,
            duration=duration,
            release=release,
            environment=environment,
            user_agent=user_agent,
            ip_address=ip_address,
            errors=errors,
            user=user,
        )

    @property
    def truncated_started(self):
        # type: (...) -> datetime
        return _minute_trunc(self.started)

    def update(
        self,
        sid=None,  # type: Optional[Union[str, uuid.UUID]]
        did=None,  # type: Optional[str]
        timestamp=None,  # type: Optional[datetime]
        started=None,  # type: Optional[datetime]
        duration=None,  # type: Optional[float]
        status=None,  # type: Optional[SessionStatus]
        release=None,  # type: Optional[str]
        environment=None,  # type: Optional[str]
        user_agent=None,  # type: Optional[str]
        ip_address=None,  # type: Optional[str]
        errors=None,  # type: Optional[int]
        user=None,  # type: Optional[Any]
    ):
        # type: (...) -> None
        # If a user is supplied we pull some data form it
        if user:
            if ip_address is None:
                ip_address = user.get("ip_address")
            if did is None:
                did = user.get("id") or user.get("email") or user.get("username")

        if sid is not None:
            self.sid = _make_uuid(sid)
        if did is not None:
            self.did = str(did)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.timestamp = timestamp
        if started is not None:
            self.started = started
        if duration is not None:
            self.duration = duration
        if release is not None:
            self.release = release
        if environment is not None:
            self.environment = environment
        if ip_address is not None:
            self.ip_address = ip_address
        if user_agent is not None:
            self.user_agent = user_agent
        if errors is not None:
            self.errors = errors

        if status is not None:
            self.status = status

    def close(
        self, status=None  # type: Optional[SessionStatus]
    ):
        # type: (...) -> Any
        if status is None and self.status == "ok":
            status = "exited"
        if status is not None:
            self.update(status=status)

    def get_json_attrs(
        self, with_user_info=True  # type: Optional[bool]
    ):
        # type: (...) -> Any
        attrs = {}
        if self.release is not None:
            attrs["release"] = self.release
        if self.environment is not None:
            attrs["environment"] = self.environment
        if with_user_info:
            if self.ip_address is not None:
                attrs["ip_address"] = self.ip_address
            if self.user_agent is not None:
                attrs["user_agent"] = self.user_agent
        return attrs

    def to_json(self):
        # type: (...) -> Any
        rv = {
            "sid": str(self.sid),
            "init": True,
            "started": format_timestamp(self.started),
            "timestamp": format_timestamp(self.timestamp),
            "status": self.status,
        }  # type: Dict[str, Any]
        if self.errors:
            rv["errors"] = self.errors
        if self.did is not None:
            rv["did"] = self.did
        if self.duration is not None:
            rv["duration"] = self.duration
        attrs = self.get_json_attrs()
        if attrs:
            rv["attrs"] = attrs
        return rv
