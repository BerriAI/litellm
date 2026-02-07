"""Public API for Opik payload building."""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from litellm.integrations.opik import utils

from . import extractors, payload_builders, types


def build_opik_payload(
    kwargs: Dict[str, Any],
    response_obj: Dict[str, Any],
    start_time: datetime,
    end_time: datetime,
    project_name: str,
) -> Tuple[Optional[types.TracePayload], types.SpanPayload]:
    """
    Build Opik trace and span payloads from LiteLLM completion data.

    This is the main public API for creating Opik payloads. It:
    1. Extracts all necessary data from LiteLLM kwargs and response
    2. Decides whether to create a new trace or attach to existing
    3. Builds trace payload (if new trace)
    4. Builds span payload (always)

    Args:
        kwargs: LiteLLM kwargs containing request metadata and logging data
        response_obj: LiteLLM response object containing model response
        start_time: Request start time
        end_time: Request end time
        project_name: Default Opik project name

    Returns:
        Tuple of (optional trace payload, span payload)
        - First element is TracePayload if creating a new trace, None if attaching to existing
        - Second element is always SpanPayload
    """
    standard_logging_object = kwargs["standard_logging_object"]

    # Extract litellm params and metadata
    litellm_params = kwargs.get("litellm_params", {}) or {}
    litellm_metadata = litellm_params.get("metadata", {}) or {}
    standard_logging_metadata = standard_logging_object.get("metadata", {}) or {}

    # Extract and merge Opik metadata
    opik_metadata = extractors.extract_opik_metadata(
        litellm_metadata, standard_logging_metadata
    )

    # Extract project name
    current_project_name = opik_metadata.get("project_name", project_name)

    # Extract trace identifiers
    current_span_data = opik_metadata.get("current_span_data")
    trace_id, parent_span_id = extractors.extract_span_identifiers(current_span_data)

    # Extract tags and thread_id
    tags = extractors.extract_tags(opik_metadata, kwargs.get("custom_llm_provider"))
    thread_id = opik_metadata.get("thread_id")

    # Apply proxy header overrides
    proxy_request = litellm_params.get("proxy_server_request", {}) or {}
    proxy_headers = proxy_request.get("headers", {}) or {}
    current_project_name, tags, thread_id = extractors.apply_proxy_header_overrides(
        current_project_name, tags, thread_id, proxy_headers
    )

    # Build shared metadata
    metadata = extractors.extract_and_build_metadata(
        opik_metadata=opik_metadata,
        standard_logging_metadata=standard_logging_metadata,
        standard_logging_object=standard_logging_object,
        litellm_kwargs=kwargs,
    )

    # Get input/output data
    input_data = standard_logging_object.get("messages", {})
    output_data = standard_logging_object.get("response", {})

    # Decide whether to create a new trace or attach to existing
    trace_payload: Optional[types.TracePayload] = None
    if trace_id is None:
        trace_id = utils.create_uuid7()
        trace_payload = payload_builders.build_trace_payload(
            project_name=current_project_name,
            trace_id=trace_id,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
            tags=tags,
            thread_id=thread_id,
        )

    # Always create a span
    usage = utils.create_usage_object(response_obj["usage"])
    
    # Extract provider and cost
    provider = extractors.normalize_provider_name(kwargs.get("custom_llm_provider"))
    cost = kwargs.get("response_cost")
    
    span_payload = payload_builders.build_span_payload(
        project_name=current_project_name,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
        input_data=input_data,
        output_data=output_data,
        metadata=metadata,
        tags=tags,
        usage=usage,
        provider=provider,
        cost=cost,
    )

    return trace_payload, span_payload
