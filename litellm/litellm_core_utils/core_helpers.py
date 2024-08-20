# What is this?
## Helper utilities
import os
from typing import BinaryIO, List, Literal, Optional, Tuple

from litellm._logging import verbose_logger


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
        finish_reason == "FINISH_REASON_UNSPECIFIED" or finish_reason == "STOP"
    ):  # vertex ai - got from running `print(dir(response_obj.candidates[0].finish_reason))`: ['FINISH_REASON_UNSPECIFIED', 'MAX_TOKENS', 'OTHER', 'RECITATION', 'SAFETY', 'STOP',]
        return "stop"
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
    elif finish_reason == "content_filtered":
        return "content_filter"
    return finish_reason


def remove_index_from_tool_calls(messages, tool_calls):
    for tool_call in tool_calls:
        if "index" in tool_call:
            tool_call.pop("index")

    for message in messages:
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            for tool_call in tool_calls:
                if "index" in tool_call:
                    tool_call.pop("index")

    return


def get_litellm_metadata_from_kwargs(kwargs: dict):
    """
    Helper to get litellm metadata from all litellm request kwargs
    """
    return kwargs.get("litellm_params", {}).get("metadata", {})


# Helper functions used for OTEL logging
def _get_parent_otel_span_from_kwargs(kwargs: Optional[dict] = None):
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
    except:
        return None


def get_file_check_sum(_file: BinaryIO):
    """
    Helper to safely get file checksum - used as a cache key
    """
    try:
        file_descriptor = _file.fileno()
        file_stat = os.fstat(file_descriptor)
        file_size = str(file_stat.st_size)
        file_checksum = _file.name + file_size
        return file_checksum
    except Exception as e:
        verbose_logger.error(f"Error getting file_checksum: {(str(e))}")
        file_checksum = _file.name
        return file_checksum
    return file_checksum
