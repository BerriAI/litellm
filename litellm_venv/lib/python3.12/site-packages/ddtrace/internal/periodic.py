# -*- encoding: utf-8 -*-
import atexit
import typing  # noqa:F401

from ddtrace.internal import forksafe
from ddtrace.internal import service
from ddtrace.internal._threads import PeriodicThread
from ddtrace.internal._threads import periodic_threads


@atexit.register
def _():
    # If the interpreter is shutting down we need to make sure that the threads
    # are stopped before the runtime is marked as finalising. This is because
    # any attempt to acquire the GIL while the runtime is finalising will cause
    # the acquiring thread to be terminated with pthread_exit (on Linux). This
    # causes a SIGABRT with GCC that cannot be caught, so we need to avoid
    # getting to that stage.
    for thread in periodic_threads.values():
        thread._atexit()


@forksafe.register
def _():
    # No threads are running after a fork so we clean up the periodic threads
    for thread in periodic_threads.values():
        thread._after_fork()
    periodic_threads.clear()


class PeriodicService(service.Service):
    """A service that runs periodically."""

    def __init__(self, interval: float = 0.0) -> None:
        super().__init__()
        self._interval = interval
        self._worker: typing.Optional[PeriodicThread] = None

    @property
    def interval(self):
        # type: (...) -> float
        return self._interval

    @interval.setter
    def interval(
        self,
        value,  # type: float
    ):
        # type: (...) -> None
        self._interval = value
        # Update the interval of the PeriodicThread based on ours
        if self._worker:
            self._worker.interval = value

    def _start_service(self, *args, **kwargs):
        # type: (typing.Any, typing.Any) -> None
        """Start the periodic service."""
        self._worker = PeriodicThread(
            self.interval,
            target=self.periodic,
            name="%s:%s" % (self.__class__.__module__, self.__class__.__name__),
            on_shutdown=self.on_shutdown,
        )
        self._worker.start()

    def _stop_service(self, *args, **kwargs):
        # type: (typing.Any, typing.Any) -> None
        """Stop the periodic collector."""
        if self._worker:
            self._worker.stop()
        super(PeriodicService, self)._stop_service(*args, **kwargs)

    def join(
        self,
        timeout=None,  # type: typing.Optional[float]
    ):
        # type: (...) -> None
        if self._worker:
            self._worker.join(timeout)

    @staticmethod
    def on_shutdown():
        pass

    def periodic(self):
        # type: (...) -> None
        pass


class AwakeablePeriodicService(PeriodicService):
    """A service that runs periodically but that can also be awakened on demand."""

    def awake(self):
        # type: (...) -> None
        if self._worker:
            self._worker.awake()


class ForksafeAwakeablePeriodicService(AwakeablePeriodicService):
    """An awakeable periodic service that auto-restarts on fork."""

    def reset(self) -> None:
        """Reset the service on fork.

        Implement this to clear the service state before restarting the thread
        in the child process.
        """
        pass

    def _restart(self) -> None:
        self.reset()
        super()._start_service()

    def _start_service(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super()._start_service(*args, **kwargs)
        forksafe.register(self._restart)

    def _stop_service(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        forksafe.unregister(self._restart)
        super()._stop_service(*args, **kwargs)
