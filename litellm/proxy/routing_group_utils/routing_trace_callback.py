"""
Routing trace callback for capturing routing decisions during test/simulation.
"""

from typing import Any, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RoutingTrace


class RoutingTraceCallback(CustomLogger):
    """
    Temporary callback registered during routing group test/simulation.

    Captures which deployments were tried, outcomes, latencies, and fallback depth.
    Attach to litellm.callbacks before a call and detach after to capture traces.
    """

    def __init__(self) -> None:
        self.traces: List[RoutingTrace] = []

    def _extract_trace_from_kwargs(
        self,
        kwargs: dict,
        start_time: Any,
        end_time: Any,
        status: str,
        exception: Optional[Exception] = None,
    ) -> RoutingTrace:
        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}

        # Calculate latency
        try:
            if hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
                latency_ms = (end_time.timestamp() - start_time.timestamp()) * 1000
            else:
                latency_ms = float(end_time - start_time) * 1000
        except Exception:
            latency_ms = 0.0

        fallback_depth = metadata.get("fallback_depth", 0)
        if not isinstance(fallback_depth, int):
            fallback_depth = 0

        deployment_id = (
            metadata.get("model_id")
            or kwargs.get("model_id")
            or kwargs.get("litellm_call_id", "unknown")
        )

        return RoutingTrace(
            deployment_id=str(deployment_id),
            deployment_name=kwargs.get("model", "unknown"),
            provider=kwargs.get("custom_llm_provider")
            or litellm_params.get("custom_llm_provider", "unknown"),
            latency_ms=latency_ms,
            was_fallback=fallback_depth > 0,
            fallback_depth=fallback_depth,
            status=status,
            error_message=str(exception) if exception else None,
        )

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        trace = self._extract_trace_from_kwargs(
            kwargs, start_time, end_time, status="success"
        )
        self.traces.append(trace)
        verbose_proxy_logger.debug(
            f"RoutingTraceCallback: success on {trace.deployment_name} ({trace.latency_ms:.0f}ms)"
        )

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        exception = kwargs.get("exception")
        trace = self._extract_trace_from_kwargs(
            kwargs, start_time, end_time, status="error", exception=exception
        )
        self.traces.append(trace)
        verbose_proxy_logger.debug(
            f"RoutingTraceCallback: failure on {trace.deployment_name} - {trace.error_message}"
        )
