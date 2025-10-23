# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import copy
from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import StandardCallbackDynamicParams
import asyncio

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
                if isinstance(choice, litellm.Choices):
                    choice.message.content = "redacted-by-litellm"
                elif isinstance(choice, litellm.utils.StreamingChoices):
                    choice.delta.content = "redacted-by-litellm"
        elif hasattr(_streaming_response, "output"):
            # Handle ResponsesAPIResponse format
            for output_item in _streaming_response.output:
                if hasattr(output_item, "content") and isinstance(
                    output_item.content, list
                ):
                    for content_part in output_item.content:
                        if hasattr(content_part, "text"):
                            content_part.text = "redacted-by-litellm"

    # Redact result
    if result is not None:
        # Check if result is a coroutine, async generator, or other async object - these cannot be deepcopied
        if (asyncio.iscoroutine(result) or 
            asyncio.iscoroutinefunction(result) or
            hasattr(result, '__aiter__') or  # async generator
            hasattr(result, '__anext__')):   # async iterator
            # For async objects, return a simple redacted response without deepcopy
            return {"text": "redacted-by-litellm"}
        
        _result = copy.deepcopy(result)
        if isinstance(_result, litellm.ModelResponse):
            if hasattr(_result, "choices") and _result.choices is not None:
                for choice in _result.choices:
                    if isinstance(choice, litellm.Choices):
                        choice.message.content = "redacted-by-litellm"
                    elif isinstance(choice, litellm.utils.StreamingChoices):
                        choice.delta.content = "redacted-by-litellm"
        elif isinstance(_result, litellm.ResponsesAPIResponse):
            if hasattr(_result, "output"):
                for output_item in _result.output:
                    if hasattr(output_item, "content") and isinstance(output_item.content, list):
                        for content_part in output_item.content:
                            if hasattr(content_part, "text"):
                                content_part.text = "redacted-by-litellm"
        elif isinstance(_result, litellm.EmbeddingResponse):
            if hasattr(_result, "data") and _result.data is not None:
                _result.data = []
        else:
            return {"text": "redacted-by-litellm"}
        return _result


def should_redact_message_logging(model_call_details: dict) -> bool:
    """
    Determine if message logging should be redacted.
    """
    _request_headers = (
        model_call_details.get("litellm_params", {}).get("metadata", {}) or {}
    )

    request_headers = _request_headers.get("headers", {})

    possible_request_headers = [
        "litellm-enable-message-redaction",  # old header. maintain backwards compatibility
        "x-litellm-enable-message-redaction",  # new header
    ]

    is_redaction_enabled_via_header = False
    for header in possible_request_headers:
        if bool(request_headers.get(header, False)):
            is_redaction_enabled_via_header = True
            break

    # check if user opted out of logging message/response to callbacks
    if (
        litellm.turn_off_message_logging is not True
        and is_redaction_enabled_via_header is not True
        and _get_turn_off_message_logging_from_dynamic_params(model_call_details)
        is not True
    ):
        return False

    if request_headers and bool(
        request_headers.get("litellm-disable-message-redaction", False)
    ):
        return False

    # user has OPTED OUT of message redaction
    if _get_turn_off_message_logging_from_dynamic_params(model_call_details) is False:
        return False

    return True


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
