# What is this?
## Helper utilities
import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


def map_finish_reason(finish_reason: str) -> str:
    """
    Maps finish reasons from various AI providers (OpenAI, Vertex AI, HuggingFace, Cohere, Anthropic) to a consistent set 
    of values based on OpenAI's finish reason values.

    Provider-specific values:
    - OpenAI: 'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
    - Vertex AI: 'FINISH_REASON_UNSPECIFIED', 'STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION', 'LANGUAGE', 'OTHER', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'MALFORMED_FUNCTION_CALL'
    - HuggingFace: 'stop_sequence', 'eos_token', 'max_tokens'
    - Cohere: 'COMPLETE', 'STOP_SEQUENCE', 'MAX_TOKENS', 'TOOL_CALL', 'ERROR'
    - Anthropic: 'end_turn', 'max_tokens', 'stop_sequence', 'tool_use'

    Provider-speicific values source:
     - openai.types.chat.chat_completion.Choice.model_fields['finish_reason']
     - google.generativeai.protos.Candidate.FinishReason.__members__.keys()
     - cohere.types.ChatFinishReason
     - anthropic.types.Message.model_fields['stop_reason']
    """
    # Mapping of provider-specific finish reasons to standardized values
    finish_reason_mapping = {
        # Normal completion reasons
        "stop": "stop",
        "complete": "stop",
        "eos_token": "stop",
        "stop_sequence": "stop",
        "end_turn": "stop",
        "other": "stop",
        "finish_reason_unspecified": "stop",

        # Error completions reasons
        # Mapping to stop since our set of finish reasons doesn't include error
        "error": "stop",
        "malformed_function_call": "stop",

        # Length-related reasons
        "length": "length",
        "max_tokens": "length",
        
        # Tool/function call reasons
        "tool_calls": "tool_calls",
        "tool_use": "tool_calls",
        "function_call": "tool_calls",
        "tool_call": "tool_calls",
        
        # Content filtering/safety reasons
        "content_filter": "content_filter",
        "error_toxic": "content_filter",
        "safety": "content_filter",
        "recitation": "content_filter",
        "content_filtered": "content_filter",
        "blocklist": "content_filter",
        "prohibited_content": "content_filter",
        "spii": "content_filter",
    }
    
    return finish_reason_mapping.get(finish_reason.lower(), finish_reason)


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
