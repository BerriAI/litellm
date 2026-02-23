# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import asyncio
import copy
import inspect
from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
)
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObject,
    )

    LiteLLMLoggingObject = _LiteLLMLoggingObject
else:
    LiteLLMLoggingObject = Any


def redact_message_input_output_from_custom_logger(
    litellm_logging_obj: LiteLLMLoggingObject, result, custom_logger: CustomLogger
):
    if (
        hasattr(custom_logger, "message_logging")
        and custom_logger.message_logging is not True
    ):
        return perform_redaction(litellm_logging_obj.model_call_details, result)
    return result


def _redact_choice_content(choice):
    """Helper to redact content in a choice (message or delta)."""
    if isinstance(choice, litellm.Choices):
        choice.message.content = "redacted-by-litellm"
        if hasattr(choice.message, "reasoning_content"):
            choice.message.reasoning_content = "redacted-by-litellm"
        if hasattr(choice.message, "thinking_blocks"):
            choice.message.thinking_blocks = None
    elif isinstance(choice, litellm.utils.StreamingChoices):
        choice.delta.content = "redacted-by-litellm"
        if hasattr(choice.delta, "reasoning_content"):
            choice.delta.reasoning_content = "redacted-by-litellm"
        if hasattr(choice.delta, "thinking_blocks"):
            choice.delta.thinking_blocks = None


def _redact_responses_api_output(output_items):
    """Helper to redact ResponsesAPIResponse output items."""
    for output_item in output_items:
        if hasattr(output_item, "content") and isinstance(output_item.content, list):
            for content_part in output_item.content:
                if hasattr(content_part, "text"):
                    content_part.text = "redacted-by-litellm"
        
        # Redact reasoning items in output array
        if hasattr(output_item, "type") and output_item.type == "reasoning":
            if hasattr(output_item, "summary") and isinstance(output_item.summary, list):
                for summary_item in output_item.summary:
                    if hasattr(summary_item, "text"):
                        summary_item.text = "redacted-by-litellm"


def perform_redaction(model_call_details: dict, result):
    """
    Performs the actual redaction on the logging object and result.
    """
    # Redact model_call_details
    model_call_details["messages"] = [
        {"role": "user", "content": "redacted-by-litellm"}
    ]
    model_call_details["prompt"] = ""
    model_call_details["input"] = ""

    # Redact streaming response
    if (
        model_call_details.get("stream", False) is True
        and "complete_streaming_response" in model_call_details
    ):
        _streaming_response = model_call_details["complete_streaming_response"]
        if hasattr(_streaming_response, "choices"):
            for choice in _streaming_response.choices:
                _redact_choice_content(choice)
        elif hasattr(_streaming_response, "output"):
            _redact_responses_api_output(_streaming_response.output)
            # Redact reasoning field in ResponsesAPIResponse
            if hasattr(_streaming_response, "reasoning") and _streaming_response.reasoning is not None:
                _streaming_response.reasoning = None

    # Redact result
    if result is not None:
        # Check if result is a coroutine, async generator, or other async object - these cannot be deepcopied
        if (asyncio.iscoroutine(result) or
            inspect.iscoroutinefunction(result) or
            hasattr(result, '__aiter__') or  # async generator
            hasattr(result, '__anext__')):   # async iterator
            # For async objects, return a simple redacted response without deepcopy
            return {"text": "redacted-by-litellm"}
        
        _result = copy.deepcopy(result)
        if isinstance(_result, litellm.ModelResponse):
            if hasattr(_result, "choices") and _result.choices is not None:
                for choice in _result.choices:
                    _redact_choice_content(choice)
        elif isinstance(_result, litellm.ResponsesAPIResponse):
            if hasattr(_result, "output"):
                _redact_responses_api_output(_result.output)
            # Redact reasoning field in ResponsesAPIResponse
            if hasattr(_result, "reasoning") and _result.reasoning is not None:
                _result.reasoning = None
        elif isinstance(_result, litellm.EmbeddingResponse):
            if hasattr(_result, "data") and _result.data is not None:
                _result.data = []
        else:
            return {"text": "redacted-by-litellm"}
        return _result


def should_redact_message_logging(model_call_details: dict) -> bool:
    """
    Determine if message logging should be redacted.
    
    Priority order:
    1. Dynamic parameter (turn_off_message_logging in request)
    2. Headers (litellm-disable-message-redaction / litellm-enable-message-redaction)
    3. Global setting (litellm.turn_off_message_logging)
    """
    litellm_params = model_call_details.get("litellm_params", {})
    
    metadata_field = get_metadata_variable_name_from_kwargs(litellm_params)
    metadata = litellm_params.get(metadata_field, {})
    if not isinstance(metadata, dict):
        # Fall back: litellm_metadata was None, try metadata
        metadata = litellm_params.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Get headers from the metadata
    request_headers = metadata.get("headers", {})

    # Check for headers that explicitly control redaction
    if request_headers and bool(
        request_headers.get("litellm-disable-message-redaction", False)
    ):
        # User explicitly disabled redaction via header
        return False

    possible_enable_headers = [
        "litellm-enable-message-redaction",  # old header. maintain backwards compatibility
        "x-litellm-enable-message-redaction",  # new header
    ]

    is_redaction_enabled_via_header = False
    for header in possible_enable_headers:
        if bool(request_headers.get(header, False)):
            is_redaction_enabled_via_header = True
            break

    # Priority 1: Check dynamic parameter first (if explicitly set)
    dynamic_turn_off = _get_turn_off_message_logging_from_dynamic_params(model_call_details)
    if dynamic_turn_off is not None:
        # Dynamic parameter is explicitly set, use it
        return dynamic_turn_off
    
    # Priority 2: Check if header explicitly enables redaction
    if is_redaction_enabled_via_header:
        return True
    
    # Priority 3: Fall back to global setting
    return litellm.turn_off_message_logging is True


def redact_message_input_output_from_logging(
    model_call_details: dict, result, input: Optional[Any] = None
) -> Any:
    """
    Removes messages, prompts, input, response from logging. This modifies the data in-place
    only redacts when litellm.turn_off_message_logging == True
    """
    if should_redact_message_logging(model_call_details):
        return perform_redaction(model_call_details, result)
    return result


def _get_turn_off_message_logging_from_dynamic_params(
    model_call_details: dict,
) -> Optional[bool]:
    """
    gets the value of `turn_off_message_logging` from the dynamic params, if it exists.

    handles boolean and string values of `turn_off_message_logging`
    """
    standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = (
        model_call_details.get("standard_callback_dynamic_params", None)
    )
    if standard_callback_dynamic_params:
        _turn_off_message_logging = standard_callback_dynamic_params.get(
            "turn_off_message_logging"
        )
        if isinstance(_turn_off_message_logging, bool):
            return _turn_off_message_logging
        elif isinstance(_turn_off_message_logging, str):
            return str_to_bool(_turn_off_message_logging)
    return None


def redact_user_api_key_info(metadata: dict) -> dict:
    """
    removes any user_api_key_info before passing to logging object, if flag set

    Usage:

    SDK
    ```python
    litellm.redact_user_api_key_info = True
    ```

    PROXY:
    ```yaml
    litellm_settings:
        redact_user_api_key_info: true
    ```
    """
    if litellm.redact_user_api_key_info is not True:
        return metadata

    new_metadata = {}
    for k, v in metadata.items():
        if isinstance(k, str) and k.startswith("user_api_key"):
            pass
        else:
            new_metadata[k] = v

    return new_metadata
