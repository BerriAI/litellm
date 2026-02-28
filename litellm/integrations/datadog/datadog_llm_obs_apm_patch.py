"""
Helper for DataDogLLMObsLogger to properly correlate LLM Obs traces with APM traces.

The issue is that the default implementation uses `tracer` from 
`litellm.litellm_core_utils.dd_tracing` which is a NullTracer unless USE_DDTRACE=True is set.

This module imports directly from ddtrace to get the active APM context.
"""

from typing import Optional, Tuple


def _get_apm_trace_context() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieve the current APM trace ID and span ID directly from ddtrace.

    This bypasses the litellm.litellm_core_utils.dd_tracing module which uses
    a NullTracer by default.

    Returns:
        Tuple of (trace_id, span_id) as strings, or (None, None) if not available.
    """
    try:
        # Import directly from ddtrace, not from litellm's wrapper
        from ddtrace import tracer

        current_span = tracer.current_span()
        if current_span is not None:
            trace_id = getattr(current_span, "trace_id", None)
            span_id = getattr(current_span, "span_id", None)
            if trace_id is not None and span_id is not None:
                return str(trace_id), str(span_id)
    except ImportError:
        # ddtrace not installed
        pass
    except Exception:
        # Any other error, fail silently
        pass

    return None, None
