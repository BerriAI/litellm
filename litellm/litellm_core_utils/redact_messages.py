# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import copy
from typing import TYPE_CHECKING, Any, Optional, Union

from gotrue import model

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import (
    DynamicCallbackSettings,
    PreRedactionFields,
    StandardCallbackDynamicParams,
    StandardLoggingPayload,
)

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
        for choice in _streaming_response.choices:
            if isinstance(choice, litellm.Choices):
                choice.message.content = "redacted-by-litellm"
            elif isinstance(choice, litellm.utils.StreamingChoices):
                choice.delta.content = "redacted-by-litellm"

    # Redact result
    if result is not None and isinstance(result, litellm.ModelResponse):
        _result = copy.deepcopy(result)
        if hasattr(_result, "choices") and _result.choices is not None:
            for choice in _result.choices:
                if isinstance(choice, litellm.Choices):
                    choice.message.content = "redacted-by-litellm"
                elif isinstance(choice, litellm.utils.StreamingChoices):
                    choice.delta.content = "redacted-by-litellm"
        return _result
    else:
        return "redacted-by-litellm"


def redact_message_input_output_from_logging(
    model_call_details: dict,
    result,
    litellm_logging_obj: Optional[LiteLLMLoggingObject] = None,
    input: Optional[Any] = None,
):
    """
    Removes messages, prompts, input, response from logging. This modifies the data in-place
    only redacts when litellm.turn_off_message_logging == True
    """
    _init_pre_redaction_fields(
        litellm_logging_obj=litellm_logging_obj,
        model_call_details=model_call_details,
        result=result,
    )
    _request_headers = (
        model_call_details.get("litellm_params", {}).get("metadata", {}) or {}
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

    return perform_redaction(
        model_call_details=model_call_details,
        result=result,
    )


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


def _init_pre_redaction_fields(
    model_call_details: dict,
    result: Any,
    litellm_logging_obj: Optional[LiteLLMLoggingObject] = None,
):
    if litellm_logging_obj is None:
        return
    if (
        "messages" not in litellm_logging_obj.pre_redaction_fields
        and model_call_details.get("messages", None)
    ):
        litellm_logging_obj.pre_redaction_fields["messages"] = model_call_details.get(
            "messages", None
        )
    if (
        "input" not in litellm_logging_obj.pre_redaction_fields
        and model_call_details.get("input", None)
    ):
        litellm_logging_obj.pre_redaction_fields["input"] = model_call_details.get(
            "input", None
        )
    if (
        "prompt" not in litellm_logging_obj.pre_redaction_fields
        and model_call_details.get("prompt", None)
    ):
        litellm_logging_obj.pre_redaction_fields["prompt"] = model_call_details.get(
            "prompt", None
        )
    if "response" not in litellm_logging_obj.pre_redaction_fields and result:
        litellm_logging_obj.pre_redaction_fields["response"] = result


def _run_redaction_for_callback(
    callback: Union[CustomLogger, str],
    model_call_details: dict,
    result: Any,
    litellm_logging_obj: LiteLLMLoggingObject,
) -> Any:
    """
    Runs redaction if provided callback has it switched on
    Provides original data to callback if redaction is disabled for it

    """
    import json

    from litellm.litellm_core_utils.litellm_logging import (
        map_custom_logger_to_callback_name,
    )

    callback_name: Optional[str] = None
    litellm_callback_name = map_custom_logger_to_callback_name(callback)
    standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = (
        model_call_details.get("standard_callback_dynamic_params", None) or {}
    )
    dynamic_callback_settings: Optional[DynamicCallbackSettings] = (
        standard_callback_dynamic_params.get("dynamic_callback_settings", None) or {}
    )

    # check if callback has redaction disabled
    dynamic_settings_for_current_callback = dynamic_callback_settings.get(
        litellm_callback_name, {}
    )
    if dynamic_settings_for_current_callback:
        if (
            dynamic_settings_for_current_callback.get("turn_off_message_logging", False)
            is False
        ):
            standard_logging_payload: Optional[StandardLoggingPayload] = (
                model_call_details.get("standard_logging_object", None)
            )
            pre_redaction_fields: PreRedactionFields = (
                litellm_logging_obj.pre_redaction_fields
            )

            if standard_logging_payload:
                standard_logging_payload["messages"] = pre_redaction_fields.get(
                    "messages", None
                )
                standard_logging_payload["response"] = pre_redaction_fields.get(
                    "response", None
                )

            model_call_details["standard_logging_object"] = standard_logging_payload

        return result
    return perform_redaction(model_call_details, result)
