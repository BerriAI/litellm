"""Mapper protocol and attribute value types."""

from typing import Sequence

from typing_extensions import Protocol, runtime_checkable

from litellm.integrations.otel.model.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    MCPToolCallSpanData,
    ServiceSpanData,
)

AttrScalar = str | bool | int | float
# Mirrors ``opentelemetry.util.types.AttributeValue`` (homogeneous sequences)
# without importing the SDK, so mappers stay OTel-free.
AttrValue = (
    AttrScalar | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)
AttributeMap = dict[str, AttrValue]

# The closed set of span-data types the engine routes through the mapper chain.
# Server spans (PROXY_REQUEST + management routes) belong to the mounted FastAPI
# instrumentor, not the mapper chain.
SpanData = LLMCallSpanData | MCPToolCallSpanData | GuardrailSpanData | ServiceSpanData


@runtime_checkable
class AttributeMapper(Protocol):
    """Maps a typed span input to a flat dict of OTel span attributes.

    One method per mapper, dispatched internally on the ``data`` type. The
    engine calls this uniformly for every span kind — mappers that don't speak
    a given type return ``{}``. This is why the engine contains no attribute keys.
    """

    def map(self, data: SpanData) -> AttributeMap: ...
