from typing import Optional

import ddtrace
from ddtrace._trace.context import Context


def _wrap_submit(func, args, kwargs):
    """
    Wrap `Executor` method used to submit a work executed in another
    thread. This wrapper ensures that a new `Context` is created and
    properly propagated using an intermediate function.
    """
    # DEV: Be sure to propagate a Context and not a Span since we are crossing thread boundaries
    current_ctx: Optional[Context] = ddtrace.tracer.current_trace_context()

    # The target function can be provided as a kwarg argument "fn" or the first positional argument
    self = args[0]
    if "fn" in kwargs:
        fn = kwargs.pop("fn")
        fn_args = args[1:]
    else:
        fn, fn_args = args[1], args[2:]
    return func(self, _wrap_execution, current_ctx, fn, fn_args, kwargs)


def _wrap_execution(ctx: Optional[Context], fn, args, kwargs):
    """
    Intermediate target function that is executed in a new thread;
    it receives the original function with arguments and keyword
    arguments, including our tracing `Context`. The current context
    provider sets the Active context in a thread local storage
    variable because it's outside the asynchronous loop.
    """
    if ctx is not None:
        ddtrace.tracer.context_provider.activate(ctx)
    return fn(*args, **kwargs)
