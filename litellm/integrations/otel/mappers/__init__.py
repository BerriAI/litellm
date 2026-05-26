"""Attribute mappers: pure ``LLMCallSpanData -> {attribute key: value}`` functions.

Composition over inheritance — each observability backend ships one mapper
instead of subclassing the engine. ``GenAIMapper`` is always present so canonical
semconv emission is guaranteed; ``LegacyMapper`` adds deprecated keys during the
dual-emit window.
"""

from litellm.integrations.otel.mappers.base import (
    AttributeMap,
    AttributeMapper,
    AttrValue,
)
from litellm.integrations.otel.mappers.genai import GenAIMapper
from litellm.integrations.otel.mappers.legacy import LegacyMapper

__all__ = [
    "AttributeMap",
    "AttributeMapper",
    "AttrValue",
    "GenAIMapper",
    "LegacyMapper",
]
