"""
Monkeypatch for DataDogLLMObsLogger to properly correlate LLM Obs traces with APM traces.

The issue is that the current implementation:
1. Uses `tracer` from `litellm.litellm_core_utils.dd_tracing` which is a NullTracer unless USE_DDTRACE=True is set
2. Only sets `apm_id` but doesn't set `parent_id` to the APM span ID
3. Falls back to generating a UUID for trace_id instead of using the APM trace_id

This patch imports directly from ddtrace to get the active APM context.

Usage:
    # Apply this patch before initializing litellm callbacks
    from litellm.integrations.datadog.datadog_llm_obs_apm_patch import patch_datadog_llm_obs_apm_correlation
    patch_datadog_llm_obs_apm_correlation()

    # Then set up litellm as usual
    import litellm
    litellm.success_callback = ["datadog_llm_obs"]
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple


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


def patch_datadog_llm_obs_apm_correlation() -> Any:
    """
    Monkey-patch DataDogLLMObsLogger to properly correlate with APM traces.

    This patch modifies create_llm_obs_payload to:
    1. Use the APM trace_id instead of generating a UUID
    2. Use the APM span_id as the parent_id
    3. Set the apm_id for trace correlation

    Returns:
        The original method (for use with unpatch_datadog_llm_obs if needed).
    """
    from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
    from litellm.types.integrations.datadog_llm_obs import LLMObsPayload

    # Store reference to original method
    _original_create_llm_obs_payload = DataDogLLMObsLogger.create_llm_obs_payload

    def _patched_create_llm_obs_payload(
        self, kwargs: Dict, start_time: datetime, end_time: datetime
    ) -> LLMObsPayload:
        """
        Patched version that properly correlates with APM traces.
        """
        # Get APM trace context BEFORE calling original method
        apm_trace_id, apm_span_id = _get_apm_trace_context()

        # Call original method to get the base payload
        payload = _original_create_llm_obs_payload(self, kwargs, start_time, end_time)

        if apm_trace_id is not None and apm_span_id is not None:
            # Use APM trace_id instead of the UUID that was generated
            payload["trace_id"] = apm_trace_id

            # Use the APM span_id as the parent_id to establish the parent-child
            # relationship between the APM span and this LLM Obs span
            payload["parent_id"] = apm_span_id

            # Set the apm_id for correlation in the Datadog UI
            payload["apm_id"] = apm_trace_id

        return payload

    # Apply the patch
    DataDogLLMObsLogger.create_llm_obs_payload = _patched_create_llm_obs_payload

    return _original_create_llm_obs_payload  # Return original in case you need to restore


def patch_datadog_llm_obs_apm_correlation_preserve_litellm_trace() -> Any:
    """
    Alternative patch that preserves litellm's trace_id but still correlates with APM.

    Use this if you want to keep litellm's trace_id separate but still have
    the LLM Obs span linked to the APM trace via apm_id and parent_id.

    Returns:
        The original method (for use with unpatch_datadog_llm_obs if needed).
    """
    from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
    from litellm.types.integrations.datadog_llm_obs import LLMObsPayload

    _original_create_llm_obs_payload = DataDogLLMObsLogger.create_llm_obs_payload

    def _patched_create_llm_obs_payload(
        self, kwargs: Dict, start_time: datetime, end_time: datetime
    ) -> LLMObsPayload:
        apm_trace_id, apm_span_id = _get_apm_trace_context()
        payload = _original_create_llm_obs_payload(self, kwargs, start_time, end_time)

        if apm_trace_id is not None and apm_span_id is not None:
            # Keep litellm's trace_id but set parent and apm correlation
            payload["parent_id"] = apm_span_id
            payload["apm_id"] = apm_trace_id

        return payload

    DataDogLLMObsLogger.create_llm_obs_payload = _patched_create_llm_obs_payload

    return _original_create_llm_obs_payload


def unpatch_datadog_llm_obs(original_method: Any) -> None:
    """
    Restore the original create_llm_obs_payload method.

    Args:
        original_method: The original method returned by one of the patch functions.
    """
    from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger

    DataDogLLMObsLogger.create_llm_obs_payload = original_method
