"""Data extraction functions for Opik payload building."""

import json
from collections.abc import Mapping
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm import _logging
from litellm.litellm_core_utils.core_helpers import iter_items, require_attributes

_DECODED_JSON: Final = TypeAdapter(object)


class _SupportsGet(Protocol):
    def get(self, key: str, default: object = None, /) -> object: ...


class _SupportsCopy(Protocol):
    def copy(self) -> object: ...


class _SupportsUpdate(Protocol):
    def update(self, other: object, /) -> object: ...


class _SpanIdentifiers(Protocol):
    @property
    def trace_id(self) -> str | None: ...

    @property
    def id(self) -> str | None: ...


def _trace_and_span_ids(span: _SpanIdentifiers) -> tuple[str | None, str | None]:
    return span.trace_id, span.id


def normalize_provider_name(provider: str | None) -> str | None:
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
    provider_mapping: Final = {
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
    litellm_metadata: object,
    standard_logging_metadata: Mapping[str, object],
) -> object:
    """
    Merge Opik metadata from three sources in increasing priority order:

      1. user_api_key_auth_metadata– lowest priority (operator-level defaults)
      2. litellm_metadata (request)– overrides auth-key defaults
      3. requester_metadata – highest priority (e.g. proxy header overrides)

    Args:
        litellm_metadata: Metadata from litellm_params.mak
        standard_logging_metadata: Metadata from standard_logging_object.

    Returns:
        Merged Opik metadata dictionary.
    """
    # Start with auth-key defaults (lowest priority).
    auth_meta: Final[object] = standard_logging_metadata.get("user_api_key_auth_metadata") or {}
    auth_opik: Final[object] = require_attributes(auth_meta, _SupportsGet, "get").get("opik") or {}
    opik_meta: Final = require_attributes(auth_opik, _SupportsCopy, "copy").copy()

    # Request-level values override auth-key defaults.
    request_opik: Final[object] = require_attributes(litellm_metadata, _SupportsGet, "get").get("opik") or {}
    require_attributes(opik_meta, _SupportsUpdate, "update").update(request_opik)

    # Requester-level values win over everything else.
    requester_metadata: Final[object] = standard_logging_metadata.get("requester_metadata", {}) or {}
    requester_opik: Final[object] = require_attributes(requester_metadata, _SupportsGet, "get").get("opik", {}) or {}
    if requester_opik:
        require_attributes(opik_meta, _SupportsUpdate, "update").update(requester_opik)

    _logging.verbose_logger.debug("litellm_opik_metadata - %s", json.dumps(opik_meta, default=str))

    return opik_meta


def extract_span_identifiers(
    current_span_data: object,
) -> tuple[str | None, str | None]:
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
        return _trace_and_span_ids(require_attributes(current_span_data, _SpanIdentifiers, "trace_id", "id"))
    except AttributeError:
        _logging.verbose_logger.warning("Unexpected current_span_data format: %s", type(current_span_data))
        return None, None


def extract_tags(
    opik_metadata: Mapping[str, object],
    custom_llm_provider: str | None,
) -> list[object]:
    """
    Extract and build list of tags.

    Args:
        opik_metadata: Opik metadata dictionary
        custom_llm_provider: LLM provider name to add as tag

    Returns:
        List of tags
    """
    metadata_tags: Final[object] = opik_metadata.get("tags", [])
    tags: Final = list(iter_items(metadata_tags))

    if custom_llm_provider:
        tags.append(custom_llm_provider)

    return tags


def apply_proxy_header_overrides(
    project_name: object,
    tags: list[object],
    thread_id: object,
    proxy_headers: Mapping[str, str],
) -> tuple[object, list[object], object]:
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
                parsed_tags = _DECODED_JSON.validate_python(json.loads(value))
                if isinstance(parsed_tags, list):
                    tags.extend(parsed_tags)
            except (json.JSONDecodeError, TypeError):
                _logging.verbose_logger.warning("Failed to parse tags from header: %s", value)

    return project_name, tags, thread_id


def extract_and_build_metadata(
    opik_metadata: Mapping[str, object],
    standard_logging_metadata: Mapping[str, object],
    standard_logging_object: Mapping[str, object],
    litellm_kwargs: Mapping[str, object],
) -> dict[str, object]:
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
    metadata: Final = {k: v for k, v in opik_metadata.items() if k != "current_span_data"}
    metadata["created_from"] = "litellm"

    # Merge with standard logging metadata
    metadata.update(standard_logging_metadata)

    # Add fields from standard_logging_object
    # These come from the LiteLLM logging infrastructure
    field_mappings: Final = {
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
        metadata["response_cost_failure_debug_info"] = litellm_kwargs["response_cost_failure_debug_info"]

    return metadata
