"""
CustomLogger callback that forwards logging events to the LoggingSidecar.

When this callback is active, it intercepts the standard logging payload
and forwards it to the sidecar process instead of processing it in the
main event loop.
"""

from typing import Dict, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.logging_sidecar import get_logging_sidecar


class LoggingSidecarHook(CustomLogger):
    """
    Lightweight callback that forwards logging events to the sidecar process.

    This replaces heavy in-process callbacks (spend tracking, etc.) with a
    queue.put_nowait() call that takes <1μs.
    """

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        sidecar = get_logging_sidecar()
        if sidecar is None or not sidecar.is_running:
            return

        standard_logging_object: Optional[Dict] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_object is None:
            return

        event = {
            "type": "success",
            "standard_logging_object": standard_logging_object,
            "model": kwargs.get("model", ""),
            "response_cost": kwargs.get("response_cost", 0),
            "start_time": str(start_time),
            "end_time": str(end_time),
        }
        sidecar.enqueue(event)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        sidecar = get_logging_sidecar()
        if sidecar is None or not sidecar.is_running:
            return

        event = {
            "type": "failure",
            "model": kwargs.get("model", ""),
            "start_time": str(start_time),
            "end_time": str(end_time),
        }
        sidecar.enqueue(event)
