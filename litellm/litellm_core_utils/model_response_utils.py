"""
Utility functions for ModelResponse and ModelResponseStream objects.
"""

from typing import Any

from litellm.types.utils import Delta, ModelResponseBase, ModelResponseStream


def is_model_response_stream_empty(model_response: ModelResponseStream) -> bool:
    """
    Check if a ModelResponseStream is empty based on:
    - If finish_reason is set -> it's non empty
    - If any field in choices is set (e.g. content, tool calls, etc.) it's non empty
    - If usage exists -> it's non empty

    This function is robust and ignores fields that are always set (from ModelResponseBase)
    and checks for any meaningful content in other fields.

    Args:
        model_response: The ModelResponseStream to check

    Returns:
        bool: True if the stream is empty, False if it contains meaningful data
    """
    # Fields that are always set in ModelResponseBase and should be ignored
    # These are structural fields that don't indicate content
    BASE_FIELDS = ModelResponseBase.model_fields.keys()

    # Check if usage exists - this indicates meaningful data
    if getattr(model_response, "usage", None) is not None:
        return False

    # Check provider_specific_fields at the top level
    if (
        hasattr(model_response, "provider_specific_fields")
        and model_response.provider_specific_fields is not None
        and model_response.provider_specific_fields != {}
    ):
        return False

    # Check model_extra for dynamically added fields (this is where Pydantic stores them)
    if hasattr(model_response, "model_extra") and model_response.model_extra:
        for extra_field_name, extra_field_value in model_response.model_extra.items():
            if _has_meaningful_content(extra_field_value):
                return False

    # Check for any non-base fields that are set
    for model_response_field in model_response.model_fields.keys():
        # Skip base fields that are always set
        if model_response_field in BASE_FIELDS:
            continue

        # Skip choices - we'll handle them separately with deep inspection
        if model_response_field == "choices":
            continue

        # Check if any other field has meaningful content
        model_response_value = getattr(model_response, model_response_field, None)
        if _has_meaningful_content(model_response_value):
            return False

    # Deep check of choices for any meaningful content
    if hasattr(model_response, "choices") and model_response.choices:
        for choice in model_response.choices:
            if _is_choice_non_empty(choice):
                return False

    # If we get here, the stream is empty
    return True


def _has_meaningful_content(value: Any) -> bool:
    """
    Check if a value contains meaningful content.

    Args:
        value: The value to check

    Returns:
        bool: True if the value has meaningful content, False otherwise
    """
    if value is None:
        return False

    if isinstance(value, str):
        # Don't strip whitespace - preserve all content including newlines, spaces, etc.
        # Even pure whitespace characters like '\n' or ' ' are meaningful content
        return len(value) > 0

    if isinstance(value, (list, dict)):
        return len(value) > 0

    if isinstance(value, bool):
        return True  # Any boolean value is meaningful

    if isinstance(value, (int, float)):
        return True  # Any numeric value is meaningful

    # For other types (objects), consider them meaningful if they exist
    return True


def _is_choice_non_empty(choice: Any) -> bool:
    """
    Deep check if a choice contains any meaningful content.

    Args:
        choice: The choice object to check

    Returns:
        bool: True if the choice has meaningful content, False otherwise
    """
    # Check finish_reason
    if hasattr(choice, "finish_reason") and choice.finish_reason is not None:

        return True

    # Check logprobs
    if hasattr(choice, "logprobs") and choice.logprobs is not None:

        return True

    # Check enhancements (if present)
    if hasattr(choice, "enhancements") and choice.enhancements is not None:

        return True

    # Deep check delta object
    if hasattr(choice, "delta") and choice.delta is not None:
        if _is_delta_non_empty(choice.delta):

            return True

    # Check model_extra for dynamically added fields on the choice
    if hasattr(choice, "model_extra") and choice.model_extra:
        for extra_field_name, extra_field_value in choice.model_extra.items():
            # Skip certain structural fields that are just default/None placeholders
            if extra_field_name == "index" and extra_field_value == 0:

                continue
            if (
                extra_field_name in {"finish_reason", "logprobs"}
                and extra_field_value is None
            ):

                continue
            if extra_field_name == "delta":

                continue
            if _has_meaningful_content(extra_field_value):

                return True

    # Check for any other non-standard fields on the choice
    for attr_name in dir(choice):
        # Skip private attributes, methods, and known empty fields
        if (
            attr_name.startswith("_")
            or callable(getattr(choice, attr_name))
            or attr_name.startswith("model_")
            or attr_name
            in {
                "finish_reason",
                "index",
                "delta",
                "logprobs",
                "enhancements",
            }
        ):

            continue

        attr_value = getattr(choice, attr_name, None)
        if _has_meaningful_content(attr_value):

            return True

    return False


def _is_delta_non_empty(delta: Delta) -> bool:
    """
    Deep check if a delta object contains any meaningful content.

    Args:
        delta: The delta object to check

    Returns:
        bool: True if the delta has meaningful content, False otherwise
    """
    # Check model_extra for dynamically added fields (this is where Pydantic stores them)
    if hasattr(delta, "model_extra") and delta.model_extra:
        for extra_field_name, extra_field_value in delta.model_extra.items():
            # Even structural fields are meaningful if they have actual content
            if _has_meaningful_content(extra_field_value):

                return True

    # Check all regular attributes of the delta object
    for attr_name in dir(delta):
        # Skip private attributes, methods, and Pydantic-specific fields
        if (
            attr_name.startswith("_")
            or callable(getattr(delta, attr_name))
            or attr_name.startswith("model_")
        ):
            continue

        attr_value = getattr(delta, attr_name, None)
        if _has_meaningful_content(attr_value):

            return True

    return False
