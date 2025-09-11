import asyncio

from ddtrace._trace.provider import BaseContextProvider
from ddtrace._trace.provider import DatadogContextMixin
from ddtrace._trace.span import Span
from ddtrace.vendor.debtcollector import deprecate


class AsyncioContextProvider(BaseContextProvider, DatadogContextMixin):
    """Manages the active context for asyncio execution. Framework
    instrumentation that is built on top of the ``asyncio`` library, should
    use this provider when contextvars are not available (Python versions
    less than 3.7).

    This Context Provider inherits from ``DefaultContextProvider`` because
    it uses a thread-local storage when the ``Context`` is propagated to
    a different thread, than the one that is running the async loop.
    """

    # Task attribute used to set/get the context
    _CONTEXT_ATTR = "__datadog_context"

    def __init__(self) -> None:
        deprecate(
            "The `ddtrace.contrib.internal.asyncio.AsyncioContextProvider` class is deprecated."
            " Use `ddtrace.DefaultContextProvider` instead.",
            version="3.0.0",
        )
        super().__init__()

    def activate(self, context, loop=None):
        """Sets the scoped ``Context`` for the current running ``Task``."""
        loop = self._get_loop(loop)
        if not loop:
            super(AsyncioContextProvider, self).activate(context)
            return context

        # the current unit of work (if tasks are used)
        task = asyncio.Task.current_task(loop=loop)
        if task:
            setattr(task, self._CONTEXT_ATTR, context)
        return context

    def _get_loop(self, loop=None):
        """Helper to try and resolve the current loop"""
        try:
            return loop or asyncio.get_event_loop()
        except RuntimeError:
            # Detects if a loop is available in the current thread;
            # DEV: This happens when a new thread is created from the out that is running the async loop
            # DEV: It's possible that a different Executor is handling a different Thread that
            #      works with blocking code. In that case, we fallback to a thread-local Context.
            pass
        return None

    def _has_active_context(self, loop=None):
        """Helper to determine if we have a currently active context"""
        loop = self._get_loop(loop=loop)
        if loop is None:
            return super(AsyncioContextProvider, self)._has_active_context()

        # the current unit of work (if tasks are used)
        task = asyncio.Task.current_task(loop=loop)
        if task is None:
            return False

        ctx = getattr(task, self._CONTEXT_ATTR, None)
        return ctx is not None

    def active(self, loop=None):
        """Returns the active context for the execution."""
        loop = self._get_loop(loop=loop)
        if not loop:
            return super(AsyncioContextProvider, self).active()

        # the current unit of work (if tasks are used)
        task = asyncio.Task.current_task(loop=loop)
        if task is None:
            return None
        ctx = getattr(task, self._CONTEXT_ATTR, None)
        if isinstance(ctx, Span):
            return self._update_active(ctx)
        return ctx
