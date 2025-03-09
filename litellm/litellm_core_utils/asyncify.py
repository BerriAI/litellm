import asyncio
import functools
from typing import Awaitable, Callable, Optional

import anyio
import anyio.to_thread
from typing_extensions import ParamSpec, TypeVar

T_ParamSpec = ParamSpec("T_ParamSpec")
T_Retval = TypeVar("T_Retval")


def function_has_argument(function: Callable, arg_name: str) -> bool:
    """Helper function to check if a function has a specific argument."""
    import inspect

    signature = inspect.signature(function)
    return arg_name in signature.parameters


def asyncify(
    function: Callable[T_ParamSpec, T_Retval],
    *,
    cancellable: bool = False,
    limiter: Optional[anyio.CapacityLimiter] = None,
) -> Callable[T_ParamSpec, Awaitable[T_Retval]]:
    """
    Take a blocking function and create an async one that receives the same
    positional and keyword arguments, and that when called, calls the original function
    in a worker thread using `anyio.to_thread.run_sync()`.

    If the `cancellable` option is enabled and the task waiting for its completion is
    cancelled, the thread will still run its course but its return value (or any raised
    exception) will be ignored.

    ## Arguments
    - `function`: a blocking regular callable (e.g. a function)
    - `cancellable`: `True` to allow cancellation of the operation
    - `limiter`: capacity limiter to use to limit the total amount of threads running
        (if omitted, the default limiter is used)

    ## Return
    An async function that takes the same positional and keyword arguments as the
    original one, that when called runs the same original function in a thread worker
    and returns the result.
    """

    async def wrapper(
        *args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs
    ) -> T_Retval:
        partial_f = functools.partial(function, *args, **kwargs)

        # In `v4.1.0` anyio added the `abandon_on_cancel` argument and deprecated the old
        # `cancellable` argument, so we need to use the new `abandon_on_cancel` to avoid
        # surfacing deprecation warnings.
        if function_has_argument(anyio.to_thread.run_sync, "abandon_on_cancel"):
            return await anyio.to_thread.run_sync(
                partial_f,
                abandon_on_cancel=cancellable,
                limiter=limiter,
            )

        return await anyio.to_thread.run_sync(
            partial_f,
            cancellable=cancellable,
            limiter=limiter,
        )

    return wrapper


def run_async_function(async_function, *args, **kwargs):
    """
    Helper utility to run an async function in a sync context.
    Handles the case where there is an existing event loop running.

    Args:
        async_function (Callable): The async function to run
        *args: Positional arguments to pass to the async function
        **kwargs: Keyword arguments to pass to the async function

    Returns:
        The result of the async function execution

    Example:
        ```python
        async def my_async_func(x, y):
            return x + y

        result = run_async_function(my_async_func, 1, 2)
        ```
    """
    from concurrent.futures import ThreadPoolExecutor

    def run_in_new_loop():
        """Run the coroutine in a new event loop within this thread."""
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(async_function(*args, **kwargs))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    try:
        # First, try to get the current event loop
        _ = asyncio.get_running_loop()
        # If we're already in an event loop, run in a separate thread
        # to avoid nested event loop issues
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()

    except RuntimeError:
        # No running event loop, we can safely run in this thread
        return run_in_new_loop()
