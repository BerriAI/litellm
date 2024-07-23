# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import copy
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObject,
    )

    LiteLLMLoggingObject = _LiteLLMLoggingObject
else:
    LiteLLMLoggingObject = Any


def redact_message_input_output_from_logging(
    litellm_logging_obj: LiteLLMLoggingObject, result
):
    """
    Removes messages, prompts, input, response from logging. This modifies the data in-place
    only redacts when litellm.turn_off_message_logging == True
    """
    _request_headers = (
        litellm_logging_obj.model_call_details.get("litellm_params", {}).get(
            "metadata", {}
        )
        or {}
    )

    request_headers = _request_headers.get("headers", {})

    # check if user opted out of logging message/response to callbacks
    if (
        litellm.turn_off_message_logging is not True
        and request_headers.get("litellm-enable-message-redaction", False) is not True
    ):
        return result

    if request_headers and request_headers.get(
        "litellm-disable-message-redaction", False
    ):
        return result

    # remove messages, prompts, input, response from logging
    litellm_logging_obj.model_call_details["messages"] = [
        {"role": "user", "content": "redacted-by-litellm"}
    ]
    litellm_logging_obj.model_call_details["prompt"] = ""
    litellm_logging_obj.model_call_details["input"] = ""

    # response cleaning
    # ChatCompletion Responses
    if (
        litellm_logging_obj.stream is True
        and "complete_streaming_response" in litellm_logging_obj.model_call_details
    ):
        _streaming_response = litellm_logging_obj.model_call_details[
            "complete_streaming_response"
        ]
        for choice in _streaming_response.choices:
            if isinstance(choice, litellm.Choices):
                choice.message.content = "redacted-by-litellm"
            elif isinstance(choice, litellm.utils.StreamingChoices):
                choice.delta.content = "redacted-by-litellm"
    else:
        if result is not None:
            if isinstance(result, litellm.ModelResponse):
                # only deep copy litellm.ModelResponse
                _result = copy.deepcopy(result)
                if hasattr(_result, "choices") and _result.choices is not None:
                    for choice in _result.choices:
                        if isinstance(choice, litellm.Choices):
                            choice.message.content = "redacted-by-litellm"
                        elif isinstance(choice, litellm.utils.StreamingChoices):
                            choice.delta.content = "redacted-by-litellm"

                return _result

    # by default return result
    return result


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
