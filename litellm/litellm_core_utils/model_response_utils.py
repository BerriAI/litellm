"""
Utility functions for ModelResponse and ModelResponseStream objects.
"""

from collections.abc import Mapping
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.types.utils import Delta, ModelResponseBase, ModelResponseStream, StreamingChoices


class _AttributeView(TypedDict):
    value: ReadOnly[object]


def _attribute_of(source: object, name: str) -> object:
    attribute: Final[_AttributeView] = {"value": getattr(source, name)}
    return attribute["value"]


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
    BASE_FIELDS: Final = ModelResponseBase.model_fields.keys()

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
    stream_extra_fields: Final[Mapping[str, object]] = model_response.model_extra or {}
    for extra_field_value in stream_extra_fields.values():
        if _has_meaningful_content(extra_field_value):
            return False

    # Check for any non-base fields that are set
    # Access model_fields on the class, not the instance, to avoid Pydantic 2.11+ deprecation warnings
    for model_response_field in type(model_response).model_fields:
        # Skip base fields that are always set
        if model_response_field in BASE_FIELDS:
            continue

        # Skip choices - we'll handle them separately with deep inspection
        if model_response_field == "choices":
            continue

        # Check if any other field has meaningful content
        model_response_value: object = getattr(model_response, model_response_field, None)
        if _has_meaningful_content(model_response_value):
            return False

    # Deep check of choices for any meaningful content
    if hasattr(model_response, "choices") and model_response.choices:
        for choice in model_response.choices:
            if _is_choice_non_empty(choice):
                return False

    # If we get here, the stream is empty
    return True


def _has_meaningful_content(value: object) -> bool:
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


def _is_choice_non_empty(choice: StreamingChoices) -> bool:
    """
    Deep check if a choice contains any meaningful content.

    Args:
        choice: The choice object to check

    Returns:
        bool: True if the choice has meaningful content, False otherwise
    """
    # Check finish_reason
    if getattr(choice, "finish_reason", None) is not None:
        return True

    # Check logprobs
    if getattr(choice, "logprobs", None) is not None:
        return True

    # Check enhancements (if present)
    if getattr(choice, "enhancements", None) is not None:
        return True

    # Deep check delta object
    choice_delta: Final[Delta | None] = getattr(choice, "delta", None)
    if choice_delta is not None and _is_delta_non_empty(choice_delta):
        return True

    # Check model_extra for dynamically added fields on the choice
    choice_extra_fields: Final[Mapping[str, object]] = choice.model_extra or {}
    for extra_field_name, extra_field_value in choice_extra_fields.items():
        if extra_field_name == "index" and extra_field_value == 0:
            continue
        if extra_field_name in {"finish_reason", "logprobs"} and extra_field_value is None:
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
            or callable(_attribute_of(choice, attr_name))
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

        choice_attr_value: object = getattr(choice, attr_name, None)
        if _has_meaningful_content(choice_attr_value):
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
    delta_extra_fields: Final[Mapping[str, object]] = delta.model_extra or {}
    for extra_field_value in delta_extra_fields.values():
        if _has_meaningful_content(extra_field_value):
            return True

    # Check all regular attributes of the delta object
    for attr_name in dir(delta):
        # Skip private attributes, methods, and Pydantic-specific fields
        if attr_name.startswith("_") or callable(_attribute_of(delta, attr_name)) or attr_name.startswith("model_"):
            continue

        delta_attr_value: object = getattr(delta, attr_name, None)
        if _has_meaningful_content(delta_attr_value):
            return True

    return False
