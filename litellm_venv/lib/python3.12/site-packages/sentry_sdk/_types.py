from typing import TYPE_CHECKING, TypeVar, Union


# Re-exported for compat, since code out there in the wild might use this variable.
MYPY = TYPE_CHECKING


SENSITIVE_DATA_SUBSTITUTE = "[Filtered]"


class AnnotatedValue:
    """
    Meta information for a data field in the event payload.
    This is to tell Relay that we have tampered with the fields value.
    See:
    https://github.com/getsentry/relay/blob/be12cd49a0f06ea932ed9b9f93a655de5d6ad6d1/relay-general/src/types/meta.rs#L407-L423
    """

    __slots__ = ("value", "metadata")

    def __init__(self, value, metadata):
        # type: (Optional[Any], Dict[str, Any]) -> None
        self.value = value
        self.metadata = metadata

    def __eq__(self, other):
        # type: (Any) -> bool
        if not isinstance(other, AnnotatedValue):
            return False

        return self.value == other.value and self.metadata == other.metadata

    @classmethod
    def removed_because_raw_data(cls):
        # type: () -> AnnotatedValue
        """The value was removed because it could not be parsed. This is done for request body values that are not json nor a form."""
        return AnnotatedValue(
            value="",
            metadata={
                "rem": [  # Remark
                    [
                        "!raw",  # Unparsable raw data
                        "x",  # The fields original value was removed
                    ]
                ]
            },
        )

    @classmethod
    def removed_because_over_size_limit(cls):
        # type: () -> AnnotatedValue
        """The actual value was removed because the size of the field exceeded the configured maximum size (specified with the max_request_body_size sdk option)"""
        return AnnotatedValue(
            value="",
            metadata={
                "rem": [  # Remark
                    [
                        "!config",  # Because of configured maximum size
                        "x",  # The fields original value was removed
                    ]
                ]
            },
        )

    @classmethod
    def substituted_because_contains_sensitive_data(cls):
        # type: () -> AnnotatedValue
        """The actual value was removed because it contained sensitive information."""
        return AnnotatedValue(
            value=SENSITIVE_DATA_SUBSTITUTE,
            metadata={
                "rem": [  # Remark
                    [
                        "!config",  # Because of SDK configuration (in this case the config is the hard coded removal of certain django cookies)
                        "s",  # The fields original value was substituted
                    ]
                ]
            },
        )


T = TypeVar("T")
Annotated = Union[AnnotatedValue, T]


if TYPE_CHECKING:
    from collections.abc import Container, MutableMapping, Sequence

    from datetime import datetime

    from types import TracebackType
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import Mapping
    from typing import NotRequired
    from typing import Optional
    from typing import Tuple
    from typing import Type
    from typing_extensions import Literal, TypedDict

    class SDKInfo(TypedDict):
        name: str
        version: str
        packages: Sequence[Mapping[str, str]]

    # "critical" is an alias of "fatal" recognized by Relay
    LogLevelStr = Literal["fatal", "critical", "error", "warning", "info", "debug"]

    DurationUnit = Literal[
        "nanosecond",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
    ]

    InformationUnit = Literal[
        "bit",
        "byte",
        "kilobyte",
        "kibibyte",
        "megabyte",
        "mebibyte",
        "gigabyte",
        "gibibyte",
        "terabyte",
        "tebibyte",
        "petabyte",
        "pebibyte",
        "exabyte",
        "exbibyte",
    ]

    FractionUnit = Literal["ratio", "percent"]
    MeasurementUnit = Union[DurationUnit, InformationUnit, FractionUnit, str]

    MeasurementValue = TypedDict(
        "MeasurementValue",
        {
            "value": float,
            "unit": NotRequired[Optional[MeasurementUnit]],
        },
    )

    Event = TypedDict(
        "Event",
        {
            "breadcrumbs": dict[
                Literal["values"], list[dict[str, Any]]
            ],  # TODO: We can expand on this type
            "check_in_id": str,
            "contexts": dict[str, dict[str, object]],
            "dist": str,
            "duration": Optional[float],
            "environment": str,
            "errors": list[dict[str, Any]],  # TODO: We can expand on this type
            "event_id": str,
            "exception": dict[
                Literal["values"], list[dict[str, Any]]
            ],  # TODO: We can expand on this type
            "extra": MutableMapping[str, object],
            "fingerprint": list[str],
            "level": LogLevelStr,
            "logentry": Mapping[str, object],
            "logger": str,
            "measurements": dict[str, MeasurementValue],
            "message": str,
            "modules": dict[str, str],
            "monitor_config": Mapping[str, object],
            "monitor_slug": Optional[str],
            "platform": Literal["python"],
            "profile": object,  # Should be sentry_sdk.profiler.Profile, but we can't import that here due to circular imports
            "release": str,
            "request": dict[str, object],
            "sdk": Mapping[str, object],
            "server_name": str,
            "spans": Annotated[list[dict[str, object]]],
            "stacktrace": dict[
                str, object
            ],  # We access this key in the code, but I am unsure whether we ever set it
            "start_timestamp": datetime,
            "status": Optional[str],
            "tags": MutableMapping[
                str, str
            ],  # Tags must be less than 200 characters each
            "threads": dict[
                Literal["values"], list[dict[str, Any]]
            ],  # TODO: We can expand on this type
            "timestamp": Optional[datetime],  # Must be set before sending the event
            "transaction": str,
            "transaction_info": Mapping[str, Any],  # TODO: We can expand on this type
            "type": Literal["check_in", "transaction"],
            "user": dict[str, object],
            "_dropped_spans": int,
            "_metrics_summary": dict[str, object],
        },
        total=False,
    )

    ExcInfo = Union[
        tuple[Type[BaseException], BaseException, Optional[TracebackType]],
        tuple[None, None, None],
    ]

    Hint = Dict[str, Any]

    Breadcrumb = Dict[str, Any]
    BreadcrumbHint = Dict[str, Any]

    SamplingContext = Dict[str, Any]

    EventProcessor = Callable[[Event, Hint], Optional[Event]]
    ErrorProcessor = Callable[[Event, ExcInfo], Optional[Event]]
    BreadcrumbProcessor = Callable[[Breadcrumb, BreadcrumbHint], Optional[Breadcrumb]]
    TransactionProcessor = Callable[[Event, Hint], Optional[Event]]

    TracesSampler = Callable[[SamplingContext], Union[float, int, bool]]

    # https://github.com/python/mypy/issues/5710
    NotImplementedType = Any

    EventDataCategory = Literal[
        "default",
        "error",
        "crash",
        "transaction",
        "security",
        "attachment",
        "session",
        "internal",
        "profile",
        "profile_chunk",
        "metric_bucket",
        "monitor",
        "span",
    ]
    SessionStatus = Literal["ok", "exited", "crashed", "abnormal"]

    ContinuousProfilerMode = Literal["thread", "gevent", "unknown"]
    ProfilerMode = Union[ContinuousProfilerMode, Literal["sleep"]]

    # Type of the metric.
    MetricType = Literal["d", "s", "g", "c"]

    # Value of the metric.
    MetricValue = Union[int, float, str]

    # Internal representation of tags as a tuple of tuples (this is done in order to allow for the same key to exist
    # multiple times).
    MetricTagsInternal = Tuple[Tuple[str, str], ...]

    # External representation of tags as a dictionary.
    MetricTagValue = Union[str, int, float, None]
    MetricTags = Mapping[str, MetricTagValue]

    # Value inside the generator for the metric value.
    FlushedMetricValue = Union[int, float]

    BucketKey = Tuple[MetricType, str, MeasurementUnit, MetricTagsInternal]
    MetricMetaKey = Tuple[MetricType, str, MeasurementUnit]

    MonitorConfigScheduleType = Literal["crontab", "interval"]
    MonitorConfigScheduleUnit = Literal[
        "year",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",  # not supported in Sentry and will result in a warning
    ]

    MonitorConfigSchedule = TypedDict(
        "MonitorConfigSchedule",
        {
            "type": MonitorConfigScheduleType,
            "value": Union[int, str],
            "unit": MonitorConfigScheduleUnit,
        },
        total=False,
    )

    MonitorConfig = TypedDict(
        "MonitorConfig",
        {
            "schedule": MonitorConfigSchedule,
            "timezone": str,
            "checkin_margin": int,
            "max_runtime": int,
            "failure_issue_threshold": int,
            "recovery_threshold": int,
        },
        total=False,
    )

    HttpStatusCodeRange = Union[int, Container[int]]
