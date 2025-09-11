from collections import namedtuple
import typing

from ddtrace._trace import span as ddspan  # noqa:F401


_T = typing.TypeVar("_T")

DDFrame = namedtuple("DDFrame", ["file_name", "lineno", "function_name", "class_name"])
StackTraceType = typing.List[DDFrame]


class Event(object):
    """An event happening at a point in time."""

    __slots__ = ()

    @property
    def name(self):
        # type: (...) -> str
        """Name of the event."""
        return self.__class__.__name__


class TimedEvent(Event):
    """An event that has a duration."""

    __slots__ = ("duration",)

    def __init__(self, duration=None, *args, **kwargs):
        # type: (typing.Optional[int], *typing.Any, **typing.Any) -> None
        super().__init__(*args, **kwargs)
        self.duration = duration


class SampleEvent(Event):
    """An event representing a sample gathered from the system."""

    __slots__ = ("sampling_period",)

    def __init__(self, sampling_period=None, *args, **kwargs):
        # type: (typing.Optional[int], *typing.Any, **typing.Any) -> None
        super().__init__(*args, **kwargs)
        self.sampling_period = sampling_period


class StackBasedEvent(SampleEvent):
    __slots__ = (
        "thread_id",
        "thread_name",
        "thread_native_id",
        "task_id",
        "task_name",
        "frames",
        "nframes",
        "local_root_span_id",
        "span_id",
        "trace_type",
        "trace_resource_container",
    )

    def __init__(
        self,
        thread_id=None,
        thread_name=None,
        thread_native_id=None,
        task_id=None,
        task_name=None,
        frames=None,
        nframes=0,
        local_root_span_id=None,
        span_id=None,
        trace_type=None,
        trace_resource_container=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.thread_native_id = thread_native_id
        self.task_id = task_id
        self.task_name = task_name
        self.frames = frames
        self.nframes = nframes
        self.local_root_span_id = local_root_span_id
        self.span_id = span_id
        self.trace_type = trace_type
        self.trace_resource_container = trace_resource_container

    def set_trace_info(
        self,
        span,  # type: typing.Optional[ddspan.Span]
        endpoint_collection_enabled,  # type: bool
    ):
        # type: (...) -> None
        if span:
            self.span_id = span.span_id
            if span._local_root is not None:
                self.local_root_span_id = span._local_root.span_id
                self.trace_type = span._local_root.span_type
                if endpoint_collection_enabled:
                    self.trace_resource_container = span._local_root._resource
