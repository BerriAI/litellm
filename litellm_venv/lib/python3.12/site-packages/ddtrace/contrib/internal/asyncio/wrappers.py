from ddtrace.contrib.internal.asyncio.compat import asyncio_current_task
from ddtrace.contrib.internal.asyncio.provider import AsyncioContextProvider


def wrapped_create_task(wrapped, instance, args, kwargs):
    """Wrapper for ``create_task(coro)`` that propagates the current active
    ``Context`` to the new ``Task``. This function is useful to connect traces
    of detached executions.

    Note: we can't just link the task contexts due to the following scenario:
        * begin task A
        * task A starts task B1..B10
        * finish task B1-B9 (B10 still on trace stack)
        * task A starts task C
        * now task C gets parented to task B10 since it's still on the stack,
          however was not actually triggered by B10
    """
    new_task = wrapped(*args, **kwargs)
    current_task = asyncio_current_task()

    ctx = getattr(current_task, AsyncioContextProvider._CONTEXT_ATTR, None)
    if ctx:
        setattr(new_task, AsyncioContextProvider._CONTEXT_ATTR, ctx)

    return new_task
