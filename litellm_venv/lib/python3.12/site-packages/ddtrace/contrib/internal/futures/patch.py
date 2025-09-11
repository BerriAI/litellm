import sys

from ddtrace.internal.wrapping import unwrap as _u
from ddtrace.internal.wrapping import wrap as _w

from .threading import _wrap_submit


def get_version():
    # type: () -> str
    return ""


def patch():
    """Enables Context Propagation between threads"""
    try:
        # Ensure that we get hold of the reloaded module if module cleanup was
        # performed.
        thread = sys.modules["concurrent.futures.thread"]
    except KeyError:
        import concurrent.futures.thread as thread

    if getattr(thread, "__datadog_patch", False):
        return
    thread.__datadog_patch = True

    _w(thread.ThreadPoolExecutor.submit, _wrap_submit)


def unpatch():
    """Disables Context Propagation between threads"""
    try:
        # Ensure that we get hold of the reloaded module if module cleanup was
        # performed.
        thread = sys.modules["concurrent.futures.thread"]
    except KeyError:
        return

    if not getattr(thread, "__datadog_patch", False):
        return
    thread.__datadog_patch = False

    _u(thread.ThreadPoolExecutor.submit, _wrap_submit)
