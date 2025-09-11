from __future__ import absolute_import

import threading
from threading import Thread
import typing  # noqa:F401

from ddtrace.internal.datadog.profiling import stack_v2
from ddtrace.settings.profiling import config

from . import _lock


class ThreadingLockAcquireEvent(_lock.LockAcquireEvent):
    """A threading.Lock has been acquired."""

    __slots__ = ()


class ThreadingLockReleaseEvent(_lock.LockReleaseEvent):
    """A threading.Lock has been released."""

    __slots__ = ()


class _ProfiledThreadingLock(_lock._ProfiledLock):
    ACQUIRE_EVENT_CLASS = ThreadingLockAcquireEvent
    RELEASE_EVENT_CLASS = ThreadingLockReleaseEvent


class ThreadingLockCollector(_lock.LockCollector):
    """Record threading.Lock usage."""

    PROFILED_LOCK_CLASS = _ProfiledThreadingLock

    def _get_patch_target(self):
        # type: (...) -> typing.Any
        return threading.Lock

    def _set_patch_target(
        self, value  # type: typing.Any
    ):
        # type: (...) -> None
        threading.Lock = value  # type: ignore[misc]


# Also patch threading.Thread so echion can track thread lifetimes
def init_stack_v2():
    if config.stack.v2_enabled and stack_v2.is_available:
        _thread_set_native_id = Thread._set_native_id
        _thread_bootstrap_inner = Thread._bootstrap_inner

        def thread_set_native_id(self, *args, **kswargs):
            _thread_set_native_id(self, *args, **kswargs)
            stack_v2.register_thread(self.ident, self.native_id, self.name)

        def thread_bootstrap_inner(self, *args, **kwargs):
            _thread_bootstrap_inner(self, *args, **kwargs)
            stack_v2.unregister_thread(self.ident)

        Thread._set_native_id = thread_set_native_id
        Thread._bootstrap_inner = thread_bootstrap_inner

        # Instrument any living threads
        for thread_id, thread in threading._active.items():
            # DEV: calling _set_native_id will register the thread with stack_v2
            # as we've already patched it.
            # Calling _set_native_id was necessary to ensure that the native_id
            # was set on the thread running in gunicorn workers. They need to be
            # updated with correct native_id so that the thread can be tracked
            # correctly in the echion stack_v2.
            thread._set_native_id()
