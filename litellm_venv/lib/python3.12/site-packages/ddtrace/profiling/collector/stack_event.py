from typing import Any  # noqa: F401
from typing import Optional

from ddtrace.profiling import event


class StackSampleEvent(event.StackBasedEvent):
    """A sample storing executions frames for a thread."""

    __slots__ = ("wall_time_ns", "cpu_time_ns")

    def __init__(self, wall_time_ns=0, cpu_time_ns=0, *args, **kwargs):
        # type: (int, int, *Any, **Any) -> None
        super().__init__(*args, **kwargs)
        # Wall clock
        self.wall_time_ns = wall_time_ns
        # CPU time in nanoseconds
        self.cpu_time_ns = cpu_time_ns


class StackExceptionSampleEvent(event.StackBasedEvent):
    """A a sample storing raised exceptions and their stack frames."""

    __slots__ = ("exc_type",)

    def __init__(self, exc_type=None, *args, **kwargs):
        # type: (Optional[str], *Any, **Any) -> None
        super().__init__(*args, **kwargs)
        self.exc_type: Optional[str] = exc_type
