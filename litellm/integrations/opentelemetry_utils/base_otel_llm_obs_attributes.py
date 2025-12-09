from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from opentelemetry.trace import Span


class BaseLLMObsOTELAttributes(ABC):
    @staticmethod
    def set_messages(span: "Span", kwargs: Dict[str, Any]):
        pass

    @staticmethod
    def set_response_output_messages(span: "Span", response_obj):
        pass


def cast_as_primitive_value_type(value) -> Union[str, bool, int, float]:
    """
    Converts a value to an OTEL-supported primitive for Arize/Phoenix observability.
    """
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def safe_set_attribute(span: "Span", key: str, value: Any):
    """
    Sets a span attribute safely with OTEL-compliant primitive typing for Arize/Phoenix.
    """
    primitive_value = cast_as_primitive_value_type(value)
    span.set_attribute(key, primitive_value)
