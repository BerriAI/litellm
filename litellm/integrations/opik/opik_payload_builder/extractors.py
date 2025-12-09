"""Data extraction functions for Opik payload building."""

import json
from typing import Any, Dict, List, Optional, Tuple

from litellm import _logging


def normalize_provider_name(provider: Optional[str]) -> Optional[str]:
    """
    Normalize LiteLLM provider names to standardized string names.
    
    Args:
        provider: LiteLLM internal provider name
        
    Returns:
        Normalized provider name or the original if no mapping exists
    """
    if provider is None:
        return None
    
    # Provider mapping to names used in Opik
    provider_mapping = {
        "openai": "openai",
        "vertex_ai-language-models": "google_vertexai",
        "gemini": "google_ai",
        "anthropic": "anthropic",
        "vertex_ai-anthropic_models": "anthropic_vertexai",
        "bedrock": "bedrock",
        "bedrock_converse": "bedrock",
        "groq": "groq",
    }
    
    return provider_mapping.get(provider, provider)


def extract_opik_metadata(
    litellm_metadata: Dict[str, Any],
    standard_logging_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract and merge Opik metadata from request and requester.

    Args:
        litellm_metadata: Metadata from litellm_params
        standard_logging_metadata: Metadata from standard_logging_object

    Returns:
        Merged Opik metadata dictionary
    """
    opik_meta = litellm_metadata.get("opik", {}).copy()

    requester_metadata = standard_logging_metadata.get("requester_metadata", {}) or {}
    requester_opik = requester_metadata.get("opik", {}) or {}
    opik_meta.update(requester_opik)

    _logging.verbose_logger.debug(
        f"litellm_opik_metadata - {json.dumps(opik_meta, default=str)}"
    )

    return opik_meta


def extract_span_identifiers(
    current_span_data: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract trace_id and parent_span_id from current_span_data.

    Args:
        current_span_data: Either dict with trace_id/id keys or Opik object

    Returns:
        Tuple of (trace_id, parent_span_id), both optional
    """
    if current_span_data is None:
        return None, None

    if isinstance(current_span_data, dict):
        return (current_span_data.get("trace_id"), current_span_data.get("id"))

    try:
        return current_span_data.trace_id, current_span_data.id
    except AttributeError:
        _logging.verbose_logger.warning(
            f"Unexpected current_span_data format: {type(current_span_data)}"
        )
        return None, None


def extract_tags(
    opik_metadata: Dict[str, Any],
    custom_llm_provider: Optional[str],
) -> List[str]:
    """
    Extract and build list of tags.

    Args:
        opik_metadata: Opik metadata dictionary
        custom_llm_provider: LLM provider name to add as tag

    Returns:
        List of tags
    """
    tags = list(opik_metadata.get("tags", []))

    if custom_llm_provider:
        tags.append(custom_llm_provider)

    return tags


def apply_proxy_header_overrides(
    project_name: str,
    tags: List[str],
    thread_id: Optional[str],
    proxy_headers: Dict[str, Any],
) -> Tuple[str, List[str], Optional[str]]:
    """
    Apply overrides from proxy request headers (opik_* prefix).

    Args:
        project_name: Current project name
        tags: Current tags list
        thread_id: Current thread ID
        proxy_headers: HTTP headers from proxy request

    Returns:
        Tuple of (project_name, tags, thread_id) with overrides applied
    """
    for key, value in proxy_headers.items():
        if not key.startswith("opik_") or not value:
            continue

        param_key = key.replace("opik_", "", 1)

        if param_key == "project_name":
            project_name = value
        elif param_key == "thread_id":
            thread_id = value
        elif param_key == "tags":
            try:
                parsed_tags = json.loads(value)
                if isinstance(parsed_tags, list):
                    tags.extend(parsed_tags)
            except (json.JSONDecodeError, TypeError):
                _logging.verbose_logger.warning(
                    f"Failed to parse tags from header: {value}"
                )

    return project_name, tags, thread_id


def extract_and_build_metadata(
    opik_metadata: Dict[str, Any],
    standard_logging_metadata: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    litellm_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the complete metadata dictionary from all available sources.

    This combines:
    - Opik-specific metadata (tags, etc.)
    - Standard logging metadata
    - Fields from standard_logging_object (model info, status, etc.)
    - Cost information from litellm_kwargs (calculated after completion)

    Args:
        opik_metadata: Opik-specific metadata from request
        standard_logging_metadata: Standard logging metadata
        standard_logging_object: Full standard logging object with call details
        litellm_kwargs: Original LiteLLM kwargs (includes response_cost)

    Returns:
        Complete metadata dictionary for trace/span
    """
    # Start with opik metadata (excluding current_span_data which is used for trace linking)
    metadata = {k: v for k, v in opik_metadata.items() if k != "current_span_data"}
    metadata["created_from"] = "litellm"

    # Merge with standard logging metadata
    metadata.update(standard_logging_metadata)

    # Add fields from standard_logging_object
    # These come from the LiteLLM logging infrastructure
    field_mappings = {
        "call_type": "type",
        "status": "status",
        "model": "model",
        "model_id": "model_id",
        "model_group": "model_group",
        "api_base": "api_base",
        "cache_hit": "cache_hit",
        "saved_cache_cost": "saved_cache_cost",
        "error_str": "error_str",
        "model_parameters": "model_parameters",
        "hidden_params": "hidden_params",
        "model_map_information": "model_map_information",
    }

    for source_key, dest_key in field_mappings.items():
        if source_key in standard_logging_object:
            metadata[dest_key] = standard_logging_object[source_key]

    # Add cost information
    # response_cost is calculated by LiteLLM after completion and added to kwargs
    # See: litellm/litellm_core_utils/llm_response_utils/response_metadata.py
    if "response_cost" in litellm_kwargs:
        metadata["cost"] = {
            "total_tokens": litellm_kwargs["response_cost"],
            "currency": "USD",
        }

    # Add debug info if cost calculation failed
    if "response_cost_failure_debug_info" in litellm_kwargs:
        metadata["response_cost_failure_debug_info"] = litellm_kwargs[
            "response_cost_failure_debug_info"
        ]

    return metadata
