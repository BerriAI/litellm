from abc import ABC
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from opentelemetry.trace import Span


class BaseLLMObsOTELAttributes(ABC):
    @staticmethod
    def set_messages(span: "Span", kwargs: dict[str, Any]):
        pass

    @staticmethod
    def set_response_output_messages(span: "Span", response_obj):
        pass


def cast_as_primitive_value_type(value) -> str | bool | int | float:
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


def is_recording(span: "Span | None") -> bool:
    """
    Whether ``span`` is a span that still accepts writes.

    Caller-owned parent spans (a web framework's server span, for instance)
    routinely end before LiteLLM's async logging handlers run, and the OTel SDK
    drops every write to an ended span with a warning.
    """
    if span is None or not hasattr(span, "is_recording"):
        return False
    return bool(span.is_recording())


def safe_set_attribute(span: "Span", key: str, value: Any):
    """
    Sets a span attribute safely with OTEL-compliant primitive typing for Arize/Phoenix.
    """
    if not is_recording(span):
        return
    primitive_value: Final = cast_as_primitive_value_type(value)
    span.set_attribute(key, primitive_value)
