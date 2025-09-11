import typing  # noqa:F401

from .. import collector
from . import _lock


class AsyncioLockAcquireEvent(_lock.LockAcquireEvent):
    """An asyncio.Lock has been acquired."""

    __slots__ = ()


class AsyncioLockReleaseEvent(_lock.LockReleaseEvent):
    """An asyncio.Lock has been released."""

    __slots__ = ()


class _ProfiledAsyncioLock(_lock._ProfiledLock):
    ACQUIRE_EVENT_CLASS = AsyncioLockAcquireEvent
    RELEASE_EVENT_CLASS = AsyncioLockReleaseEvent


class AsyncioLockCollector(_lock.LockCollector):
    """Record asyncio.Lock usage."""

    PROFILED_LOCK_CLASS = _ProfiledAsyncioLock

    def _start_service(self):
        # type: (...) -> None
        """Start collecting lock usage."""
        try:
            import asyncio
        except ImportError as e:
            raise collector.CollectorUnavailable(e)
        self._asyncio_module = asyncio
        return super(AsyncioLockCollector, self)._start_service()

    def _get_patch_target(self):
        # type: (...) -> typing.Any
        return self._asyncio_module.Lock

    def _set_patch_target(
        self, value  # type: typing.Any
    ):
        # type: (...) -> None
        self._asyncio_module.Lock = value  # type: ignore[misc]
