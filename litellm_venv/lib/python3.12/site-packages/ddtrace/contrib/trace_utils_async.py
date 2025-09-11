"""
async tracing utils

Note that this module should only be imported in Python 3.5+.
"""
from ddtrace import Pin
from ddtrace.internal.logger import get_logger


log = get_logger(__name__)


def with_traced_module(func):
    """Async version of trace_utils.with_traced_module.
    Usage::

        @with_traced_module
        async def my_traced_wrapper(django, pin, func, instance, args, kwargs):
            # Do tracing stuff
            pass

        def patch():
            import django
            wrap(django.somefunc, my_traced_wrapper(django))
    """

    def with_mod(mod):
        async def wrapper(wrapped, instance, args, kwargs):
            pin = Pin._find(instance, mod)
            if pin and not pin.enabled():
                return await wrapped(*args, **kwargs)
            elif not pin:
                log.debug("Pin not found for traced method %r", wrapped)
                return await wrapped(*args, **kwargs)
            return await func(mod, pin, wrapped, instance, args, kwargs)

        return wrapper

    return with_mod
