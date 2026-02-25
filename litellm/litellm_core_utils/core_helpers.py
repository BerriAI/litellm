# What is this?
## Helper utilities
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.types.utils import ModelResponseStream

    Span = Union[_Span, Any]
else:
    Span = Any


def safe_divide_seconds(
    seconds: float, denominator: float, default: Optional[float] = None
) -> Optional[float]:
    """
    Safely divide seconds by denominator, handling zero division.

    Args:
        seconds: Time duration in seconds
        denominator: The divisor (e.g., number of tokens)
        default: Value to return if division by zero (defaults to None)

    Returns:
        The result of the division as a float (seconds per unit), or default if denominator is zero
    """
    if denominator <= 0:
        return default

    return float(seconds / denominator)


def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: Union[int, float] = 0,
) -> Union[int, float]:
    """
    Safely divide two numbers, returning a default value if denominator is zero.

    Args:
        numerator: The number to divide
        denominator: The number to divide by
        default: Value to return if denominator is zero (defaults to 0)

    Returns:
        The result of numerator/denominator, or default if denominator is zero
    """
    if denominator == 0:
        return default
    return numerator / denominator


def map_finish_reason(
    finish_reason: str,
):  # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    # anthropic mapping
    if finish_reason == "stop_sequence":
        return "stop"
    # cohere mapping - https://docs.cohere.com/reference/generate
    elif finish_reason == "COMPLETE":
        return "stop"
    elif finish_reason == "MAX_TOKENS":  # cohere + vertex ai
        return "length"
    elif finish_reason == "ERROR_TOXIC":
        return "content_filter"
    elif (
        finish_reason == "ERROR"
    ):  # openai currently doesn't support an 'error' finish reason
        return "stop"
    # huggingface mapping https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate_stream
    elif finish_reason == "eos_token" or finish_reason == "stop_sequence":
        return "stop"
    elif (
        finish_reason == "FINISH_REASON_UNSPECIFIED"
    ):  # vertex ai - got from running `print(dir(response_obj.candidates[0].finish_reason))`: ['FINISH_REASON_UNSPECIFIED', 'MAX_TOKENS', 'OTHER', 'RECITATION', 'SAFETY', 'STOP',]
        return "finish_reason_unspecified"
    elif finish_reason == "MALFORMED_FUNCTION_CALL":
        return "malformed_function_call"
    elif finish_reason == "SAFETY" or finish_reason == "RECITATION":  # vertex ai
        return "content_filter"
    elif finish_reason == "STOP":  # vertex ai
        return "stop"
    elif finish_reason == "end_turn" or finish_reason == "stop_sequence":  # anthropic
        return "stop"
    elif finish_reason == "max_tokens":  # anthropic
        return "length"
    elif finish_reason == "tool_use":  # anthropic
        return "tool_calls"
    elif finish_reason == "compaction":
        return "length"
    return finish_reason


def remove_index_from_tool_calls(
    messages: Optional[List[AllMessageValues]],
):
    if messages is not None:
        for message in messages:
            _tool_calls = message.get("tool_calls")
            if _tool_calls is not None and isinstance(_tool_calls, list):
                for tool_call in _tool_calls:
                    if (
                        isinstance(tool_call, dict) and "index" in tool_call
                    ):  # Type guard to ensure it's a dict
                        tool_call.pop("index", None)

    return


def remove_items_at_indices(items: Optional[List[Any]], indices: Iterable[int]) -> None:
    """Remove items from a list in-place by index"""
    if items is None:
        return
    for index in sorted(set(indices), reverse=True):
        if 0 <= index < len(items):
            items.pop(index)


def add_missing_spend_metadata_to_litellm_metadata(
    litellm_metadata: dict, metadata: dict
) -> dict:
    """
    Helper to get litellm metadata for spend tracking

    PATCH for issue where both `litellm_metadata` and `metadata` are present in the kwargs
    and user_api_key values are in 'metadata'.
    """
    potential_spend_tracking_metadata_substring = "user_api_key"
    for key, value in metadata.items():
        if potential_spend_tracking_metadata_substring in key:
            litellm_metadata[key] = value
    return litellm_metadata


def get_metadata_variable_name_from_kwargs(
    kwargs: dict,
) -> Literal["metadata", "litellm_metadata"]:
    """
    Helper to return what the "metadata" field should be called in the request data

    - New endpoints return `litellm_metadata`
    - Old endpoints return `metadata`

    Context:
    - LiteLLM used `metadata` as an internal field for storing metadata
    - OpenAI then started using this field for their metadata
    - LiteLLM is now moving to using `litellm_metadata` for our metadata
    """
    return "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"


def get_litellm_metadata_from_kwargs(kwargs: dict):
    """
    Helper to get litellm metadata from all litellm request kwargs

    Return `litellm_metadata` if it exists, otherwise return `metadata`
    """
    litellm_params = kwargs.get("litellm_params", {})
    if litellm_params:
        metadata = litellm_params.get("metadata", {})
        litellm_metadata = litellm_params.get("litellm_metadata", {})
        if litellm_metadata and metadata:
            litellm_metadata = add_missing_spend_metadata_to_litellm_metadata(
                litellm_metadata, metadata
            )
        if litellm_metadata:
            return litellm_metadata
        elif metadata:
            return metadata

    return {}


def reconstruct_model_name(
    model_name: str,
    custom_llm_provider: Optional[str],
    metadata: dict,
) -> str:
    """Reconstruct full model name with provider prefix for logging."""
    # Check if deployment model name from router metadata is available (has original prefix)
    deployment_model_name = metadata.get("deployment")
    if deployment_model_name and "/" in deployment_model_name:
        # Use the deployment model name which preserves the original provider prefix
        return deployment_model_name
    elif custom_llm_provider and model_name and "/" not in model_name:
        # Only add prefix for Bedrock (not for direct Anthropic API)
        # This ensures Bedrock models get the prefix while direct Anthropic models don't
        if custom_llm_provider == "bedrock":
            return f"{custom_llm_provider}/{model_name}"
    return model_name


# Helper functions used for OTEL logging
def _get_parent_otel_span_from_kwargs(
    kwargs: Optional[dict] = None,
) -> Union[Span, None]:
    try:
        if kwargs is None:
            return None
        litellm_params = kwargs.get("litellm_params")
        _metadata = kwargs.get("metadata") or {}
        if "litellm_parent_otel_span" in _metadata:
            return _metadata["litellm_parent_otel_span"]
        elif (
            litellm_params is not None
            and litellm_params.get("metadata") is not None
            and "litellm_parent_otel_span" in litellm_params.get("metadata", {})
        ):
            return litellm_params["metadata"]["litellm_parent_otel_span"]
        elif "litellm_parent_otel_span" in kwargs:
            return kwargs["litellm_parent_otel_span"]
        return None
    except Exception as e:
        verbose_logger.exception(
            "Error in _get_parent_otel_span_from_kwargs: " + str(e)
        )
        return None


def process_response_headers(response_headers: Union[httpx.Headers, dict]) -> dict:
    from litellm.types.utils import OPENAI_RESPONSE_HEADERS

    openai_headers = {}
    processed_headers = {}
    additional_headers = {}

    for k, v in response_headers.items():
        if k in OPENAI_RESPONSE_HEADERS:  # return openai-compatible headers
            openai_headers[k] = v
        if k.startswith(
            "llm_provider-"
        ):  # return raw provider headers (incl. openai-compatible ones)
            processed_headers[k] = v
        else:
            additional_headers["{}-{}".format("llm_provider", k)] = v

    additional_headers = {
        **openai_headers,
        **processed_headers,
        **additional_headers,
    }
    return additional_headers


def preserve_upstream_non_openai_attributes(
    model_response: "ModelResponseStream", original_chunk: "ModelResponseStream"
):
    """
    Preserve non-OpenAI attributes from the original chunk.
    """
    # Access model_fields on the class, not the instance, to avoid Pydantic 2.11+ deprecation warnings
    expected_keys = set(type(model_response).model_fields.keys()).union({"usage"})
    for key, value in original_chunk.model_dump().items():
        if key not in expected_keys:
            setattr(model_response, key, value)


def safe_deep_copy(data):
    """
    Safe Deep Copy

    The LiteLLM request may contain objects that cannot be pickled/deep-copied
    (e.g., tracing spans, locks, clients).

    This helper deep-copies each top-level key independently; on failure keeps
    original ref
    """
    import copy

    import litellm

    if litellm.safe_memory_mode is True:
        return data

    litellm_parent_otel_span: Optional[Any] = None
    # Step 1: Remove the litellm_parent_otel_span
    litellm_parent_otel_span = None
    if isinstance(data, dict):
        # remove litellm_parent_otel_span since this is not picklable
        if "metadata" in data and "litellm_parent_otel_span" in data["metadata"]:
            litellm_parent_otel_span = data["metadata"].pop("litellm_parent_otel_span")
            data["metadata"]["litellm_parent_otel_span"] = "placeholder"
        if (
            "litellm_metadata" in data
            and "litellm_parent_otel_span" in data["litellm_metadata"]
        ):
            litellm_parent_otel_span = data["litellm_metadata"].pop(
                "litellm_parent_otel_span"
            )
            data["litellm_metadata"]["litellm_parent_otel_span"] = "placeholder"

    # Step 2: Per-key deepcopy with fallback
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            try:
                new_data[k] = copy.deepcopy(v)
            except Exception:
                new_data[k] = v
    else:
        try:
            new_data = copy.deepcopy(data)
        except Exception:
            new_data = data

    # Step 3: re-add the litellm_parent_otel_span after doing a deep copy
    if isinstance(data, dict) and litellm_parent_otel_span is not None:
        if "metadata" in data and "litellm_parent_otel_span" in data["metadata"]:
            data["metadata"]["litellm_parent_otel_span"] = litellm_parent_otel_span
        if (
            "litellm_metadata" in data
            and "litellm_parent_otel_span" in data["litellm_metadata"]
        ):
            data["litellm_metadata"][
                "litellm_parent_otel_span"
            ] = litellm_parent_otel_span
    return new_data


def filter_exceptions_from_params(data: Any, max_depth: int = 20) -> Any:
    """
    Recursively filter out Exception objects and callable objects from dicts/lists.

    This is a defensive utility to prevent deepcopy failures when exception objects
    are accidentally stored in parameter dictionaries (e.g., optional_params).
    Also filters callable objects (functions) to prevent JSON serialization errors.
    Exceptions and callables should not be stored in params - this function removes them.

    Args:
        data: The data structure to filter (dict, list, or any other type)
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Filtered data structure with Exception and callable objects removed, or None if the
        entire input was an Exception or callable
    """
    if max_depth <= 0:
        return data

    # Skip exception objects
    if isinstance(data, Exception):
        return None
    # Skip callable objects (functions, methods, lambdas) but not classes (type objects)
    if callable(data) and not isinstance(data, type):
        return None
    # Skip known non-serializable object types (Logging, Router, etc.)
    obj_type_name = type(data).__name__
    if obj_type_name in ["Logging", "LiteLLMLoggingObj", "Router"]:
        return None

    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for k, v in data.items():
            # Skip exception and callable values
            if isinstance(v, Exception) or (callable(v) and not isinstance(v, type)):
                continue
            try:
                filtered = filter_exceptions_from_params(v, max_depth - 1)
                if filtered is not None:
                    result[k] = filtered
            except Exception:
                # Skip values that cause errors during filtering
                continue
        return result
    elif isinstance(data, list):
        result_list: list[Any] = []
        for item in data:
            # Skip exception and callable items
            if isinstance(item, Exception) or (
                callable(item) and not isinstance(item, type)
            ):
                continue
            try:
                filtered = filter_exceptions_from_params(item, max_depth - 1)
                if filtered is not None:
                    result_list.append(filtered)
            except Exception:
                # Skip items that cause errors during filtering
                continue
        return result_list
    else:
        return data


def filter_internal_params(
    data: dict, additional_internal_params: Optional[set] = None
) -> dict:
    """
    Filter out LiteLLM internal parameters that shouldn't be sent to provider APIs.

    This removes internal/MCP-related parameters that are used by LiteLLM internally
    but should not be included in API requests to providers.

    Args:
        data: Dictionary of parameters to filter
        additional_internal_params: Optional set of additional internal parameter names to filter

    Returns:
        Filtered dictionary with internal parameters removed
    """
    if not isinstance(data, dict):
        return data

    # Known internal parameters that should never be sent to provider APIs
    internal_params = {
        "skip_mcp_handler",
        "mcp_handler_context",
        "_skip_mcp_handler",
    }

    # Add any additional internal params if provided
    if additional_internal_params:
        internal_params.update(additional_internal_params)

    # Filter out internal parameters
    return {k: v for k, v in data.items() if k not in internal_params}
