"""
Shared bidirectional mapping dictionaries between Responses API and Chat Completions formats.

Both bridges (Responses→CC and CC→Responses) import from here so the
mapping knowledge lives in one place. Asymmetries are documented inline.
"""

from typing import Any, Dict, Optional, Union

from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIStatus,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

# ---------------------------------------------------------------------------
# Status (Responses API) <-> finish_reason (Chat Completions)
# ---------------------------------------------------------------------------
# These are NOT perfect inverses. The Responses API status space is larger
# than Chat Completions finish_reason:
#   - "failed" and "cancelled" collapse to "stop" (no CC equivalent)
#   - "content_filter" maps to "incomplete" (best approximation)
#   - "tool_calls" and "function_call" map to "completed" (tools are
#     signalled separately in Responses API, not via status)
# ---------------------------------------------------------------------------

STATUS_TO_FINISH_REASON: dict = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
    "cancelled": "stop",
}

FINISH_REASON_TO_STATUS: dict = {
    "stop": "completed",
    "tool_calls": "completed",
    "function_call": "completed",
    "length": "incomplete",
    "content_filter": "incomplete",
}


# ---------------------------------------------------------------------------
# Provider-specific fields extraction
# ---------------------------------------------------------------------------
# Both bridges repeatedly normalize provider_specific_fields from objects or
# dicts into a plain dict. This helper consolidates that boilerplate.
# ---------------------------------------------------------------------------


def normalize_provider_specific_fields(obj: Any) -> Optional[Dict[str, Any]]:
    """Extract and normalize provider_specific_fields from a dict or object to a plain dict."""
    if isinstance(obj, dict):
        psf = obj.get("provider_specific_fields")
    else:
        psf = getattr(obj, "provider_specific_fields", None)

    if not psf:
        return None
    if isinstance(psf, dict):
        return psf
    return dict(psf) if hasattr(psf, "__dict__") else None


def status_to_finish_reason(status: Optional[str]) -> str:
    """Map Responses API status to Chat Completions finish_reason."""
    if not status:
        return "stop"
    return STATUS_TO_FINISH_REASON.get(status, "stop")


def finish_reason_to_status(finish_reason: Optional[str]) -> ResponsesAPIStatus:
    """Map Chat Completions finish_reason to Responses API status."""
    if finish_reason is None:
        return "completed"
    return FINISH_REASON_TO_STATUS.get(finish_reason, "completed")


# ---------------------------------------------------------------------------
# response_format (Chat Completions) <-> text.format (Responses API)
# ---------------------------------------------------------------------------
# Chat Completions nests json_schema under a "json_schema" key:
#   {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}
#
# Responses API nests everything under "format":
#   {"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": ...}}
#
# Asymmetry: "text" type produces {"format": {"type": "text"}} in the
# Responses direction, but returns None in the CC direction (it's the
# implicit default in Chat Completions).
# ---------------------------------------------------------------------------


def response_format_to_text_format(
    response_format: Union[Dict[str, Any], Any],
) -> Optional[Dict[str, Any]]:
    """
    Chat Completion response_format -> Responses API text param.

    Returns dict with "format" key, or None.
    """
    if not response_format:
        return None

    if isinstance(response_format, dict):
        format_type = response_format.get("type")

        if format_type == "json_schema":
            json_schema = response_format.get("json_schema", {})
            return {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "response_schema"),
                    "schema": json_schema.get("schema", {}),
                    "strict": json_schema.get("strict", False),
                }
            }
        elif format_type == "json_object":
            return {"format": {"type": "json_object"}}
        elif format_type == "text":
            return {"format": {"type": "text"}}

    return None


def text_format_to_response_format(
    text_param: Union[Dict[str, Any], Any],
) -> Optional[Dict[str, Any]]:
    """
    Responses API text param -> Chat Completion response_format.

    Returns None for "text" type (implicit default in CC).
    """
    if not text_param:
        return None

    if isinstance(text_param, dict):
        format_param = text_param.get("format")
        if format_param and isinstance(format_param, dict):
            format_type = format_param.get("type")

            if format_type == "json_schema":
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_param.get("name", "response_schema"),
                        "schema": format_param.get("schema", {}),
                        "strict": format_param.get("strict", False),
                    },
                }
            elif format_type == "json_object":
                return {"type": "json_object"}
            elif format_type == "text":
                return None

    return None


# ---------------------------------------------------------------------------
# Usage field mapping (Chat Completions <-> Responses API)
# ---------------------------------------------------------------------------
# Chat Completions uses prompt_tokens/completion_tokens with
# prompt_tokens_details/completion_tokens_details.
# Responses API uses input_tokens/output_tokens with
# input_tokens_details/output_tokens_details.
#
# The field names differ but the structure is the same. Both directions
# also preserve the "cost" attribute if present.
#
# Asymmetry: the CC→Responses direction defaults cached_tokens and
# reasoning_tokens to 0 when absent; the Responses→CC direction passes
# None through. This is pre-existing behavior preserved as-is.
# ---------------------------------------------------------------------------


def response_api_usage_to_chat_usage(
    usage_input: Optional[Union[dict, ResponseAPIUsage]],
) -> Usage:
    """Transform ResponseAPIUsage to Chat Completions Usage."""
    if usage_input is None:
        return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    response_api_usage: ResponseAPIUsage
    if isinstance(usage_input, dict):
        usage_input = dict(usage_input)
        total_tokens = usage_input.get("total_tokens")
        if total_tokens is None:
            input_tokens = usage_input.get("input_tokens")
            output_tokens = usage_input.get("output_tokens")
            if input_tokens is not None and output_tokens is not None:
                usage_input["total_tokens"] = input_tokens + output_tokens
        response_api_usage = ResponseAPIUsage(**usage_input)
    else:
        response_api_usage = usage_input

    prompt_tokens: int = response_api_usage.input_tokens or 0
    completion_tokens: int = response_api_usage.output_tokens or 0

    prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
    if response_api_usage.input_tokens_details:
        if isinstance(response_api_usage.input_tokens_details, dict):
            prompt_tokens_details = PromptTokensDetailsWrapper(
                **response_api_usage.input_tokens_details
            )
        else:
            prompt_tokens_details = PromptTokensDetailsWrapper(
                cached_tokens=getattr(
                    response_api_usage.input_tokens_details, "cached_tokens", None
                ),
                audio_tokens=getattr(
                    response_api_usage.input_tokens_details, "audio_tokens", None
                ),
                text_tokens=getattr(
                    response_api_usage.input_tokens_details, "text_tokens", None
                ),
                image_tokens=getattr(
                    response_api_usage.input_tokens_details, "image_tokens", None
                ),
            )

    completion_tokens_details: Optional[CompletionTokensDetailsWrapper] = None
    output_tokens_details = getattr(
        response_api_usage, "output_tokens_details", None
    )
    if output_tokens_details:
        completion_tokens_details = CompletionTokensDetailsWrapper(
            reasoning_tokens=getattr(
                output_tokens_details, "reasoning_tokens", None
            ),
            image_tokens=getattr(output_tokens_details, "image_tokens", None),
            text_tokens=getattr(output_tokens_details, "text_tokens", None),
        )

    chat_usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )

    if hasattr(response_api_usage, "cost") and response_api_usage.cost is not None:
        setattr(chat_usage, "cost", response_api_usage.cost)

    return chat_usage


def chat_usage_to_response_api_usage(
    chat_completion_response: Union[ModelResponse, Usage],
) -> ResponseAPIUsage:
    """Transform Chat Completions Usage to ResponseAPIUsage."""
    if isinstance(chat_completion_response, ModelResponse):
        usage: Optional[Usage] = getattr(chat_completion_response, "usage", None)
    else:
        usage = chat_completion_response

    if usage is None:
        return ResponseAPIUsage(input_tokens=0, output_tokens=0, total_tokens=0)

    response_usage = ResponseAPIUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )

    if hasattr(usage, "cost") and usage.cost is not None:
        setattr(response_usage, "cost", usage.cost)

    # Translate prompt_tokens_details to input_tokens_details
    if (
        hasattr(usage, "prompt_tokens_details")
        and usage.prompt_tokens_details is not None
    ):
        prompt_details = usage.prompt_tokens_details
        input_details_dict: Dict[str, int] = {}

        if (
            hasattr(prompt_details, "cached_tokens")
            and prompt_details.cached_tokens is not None
        ):
            input_details_dict["cached_tokens"] = prompt_details.cached_tokens
        else:
            input_details_dict["cached_tokens"] = 0

        if (
            hasattr(prompt_details, "text_tokens")
            and prompt_details.text_tokens is not None
        ):
            input_details_dict["text_tokens"] = prompt_details.text_tokens

        if (
            hasattr(prompt_details, "audio_tokens")
            and prompt_details.audio_tokens is not None
        ):
            input_details_dict["audio_tokens"] = prompt_details.audio_tokens

        if input_details_dict:
            response_usage.input_tokens_details = InputTokensDetails(
                **input_details_dict
            )

    # Translate completion_tokens_details to output_tokens_details
    if (
        hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details is not None
    ):
        completion_details = usage.completion_tokens_details
        output_details_dict: Dict[str, int] = {}

        if (
            hasattr(completion_details, "reasoning_tokens")
            and completion_details.reasoning_tokens is not None
        ):
            output_details_dict["reasoning_tokens"] = completion_details.reasoning_tokens
        else:
            output_details_dict["reasoning_tokens"] = 0

        if (
            hasattr(completion_details, "text_tokens")
            and completion_details.text_tokens is not None
        ):
            output_details_dict["text_tokens"] = completion_details.text_tokens

        if (
            hasattr(completion_details, "image_tokens")
            and completion_details.image_tokens is not None
        ):
            output_details_dict["image_tokens"] = completion_details.image_tokens

        if output_details_dict:
            response_usage.output_tokens_details = OutputTokensDetails(
                **output_details_dict
            )

    return response_usage
