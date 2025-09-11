from functools import wraps
from inspect import iscoroutinefunction

from sentry_sdk.crons import capture_checkin
from sentry_sdk.crons.consts import MonitorStatus
from sentry_sdk.utils import now

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType
    from typing import (
        Any,
        Optional,
        ParamSpec,
        Type,
        TypeVar,
        Union,
        cast,
        overload,
    )
    from sentry_sdk._types import MonitorConfig

    P = ParamSpec("P")
    R = TypeVar("R")


class monitor:  # noqa: N801
    """
    Decorator/context manager to capture checkin events for a monitor.

    Usage (as decorator):
    ```
    import sentry_sdk

    app = Celery()

    @app.task
    @sentry_sdk.monitor(monitor_slug='my-fancy-slug')
    def test(arg):
        print(arg)
    ```

    This does not have to be used with Celery, but if you do use it with celery,
    put the `@sentry_sdk.monitor` decorator below Celery's `@app.task` decorator.

    Usage (as context manager):
    ```
    import sentry_sdk

    def test(arg):
        with sentry_sdk.monitor(monitor_slug='my-fancy-slug'):
            print(arg)
    ```
    """

    def __init__(self, monitor_slug=None, monitor_config=None):
        # type: (Optional[str], Optional[MonitorConfig]) -> None
        self.monitor_slug = monitor_slug
        self.monitor_config = monitor_config

    def __enter__(self):
        # type: () -> None
        self.start_timestamp = now()
        self.check_in_id = capture_checkin(
            monitor_slug=self.monitor_slug,
            status=MonitorStatus.IN_PROGRESS,
            monitor_config=self.monitor_config,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        # type: (Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]) -> None
        duration_s = now() - self.start_timestamp

        if exc_type is None and exc_value is None and traceback is None:
            status = MonitorStatus.OK
        else:
            status = MonitorStatus.ERROR

        capture_checkin(
            monitor_slug=self.monitor_slug,
            check_in_id=self.check_in_id,
            status=status,
            duration=duration_s,
            monitor_config=self.monitor_config,
        )

    if TYPE_CHECKING:

        @overload
        def __call__(self, fn):
            # type: (Callable[P, Awaitable[Any]]) -> Callable[P, Awaitable[Any]]
            # Unfortunately, mypy does not give us any reliable way to type check the
            # return value of an Awaitable (i.e. async function) for this overload,
            # since calling iscouroutinefunction narrows the type to Callable[P, Awaitable[Any]].
            ...

        @overload
        def __call__(self, fn):
            # type: (Callable[P, R]) -> Callable[P, R]
            ...

    def __call__(
        self,
        fn,  # type: Union[Callable[P, R], Callable[P, Awaitable[Any]]]
    ):
        # type: (...) -> Union[Callable[P, R], Callable[P, Awaitable[Any]]]
        if iscoroutinefunction(fn):
            return self._async_wrapper(fn)

        else:
            if TYPE_CHECKING:
                fn = cast("Callable[P, R]", fn)
            return self._sync_wrapper(fn)

    def _async_wrapper(self, fn):
        # type: (Callable[P, Awaitable[Any]]) -> Callable[P, Awaitable[Any]]
        @wraps(fn)
        async def inner(*args: "P.args", **kwargs: "P.kwargs"):
            # type: (...) -> R
            with self:
                return await fn(*args, **kwargs)

        return inner

    def _sync_wrapper(self, fn):
        # type: (Callable[P, R]) -> Callable[P, R]
        @wraps(fn)
        def inner(*args: "P.args", **kwargs: "P.kwargs"):
            # type: (...) -> R
            with self:
                return fn(*args, **kwargs)

        return inner
