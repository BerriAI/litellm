import asyncio

from ddtrace.vendor.debtcollector import deprecate


if hasattr(asyncio, "current_task"):

    def asyncio_current_task():
        deprecate(
            "ddtrace.contrib.internal.asyncio.create_task(..) is deprecated. "
            "The ddtrace library fully supports propagating "
            "trace contextes to async tasks. No additional configurations are required.",
            version="3.0.0",
        )
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

else:

    def asyncio_current_task():
        deprecate(
            "ddtrace.contrib.internal.asyncio.create_task(..) is deprecated. "
            "The ddtrace library fully supports propagating "
            "trace contextes to async tasks. No additional configurations are required.",
            version="3.0.0",
        )
        return asyncio.Task.current_task()
