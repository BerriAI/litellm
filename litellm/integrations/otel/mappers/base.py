"""Mapper protocol and attribute value types."""

from typing import Dict, Sequence, Union

from typing_extensions import Protocol, runtime_checkable

from litellm.integrations.otel.payloads import LLMCallSpanData

AttrScalar = Union[str, bool, int, float]
# Mirrors ``opentelemetry.util.types.AttributeValue`` (homogeneous sequences)
# without importing the SDK, so mappers stay OTel-free.
AttrValue = Union[
    AttrScalar,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]
AttributeMap = Dict[str, AttrValue]


@runtime_checkable
class AttributeMapper(Protocol):
    """Maps a typed span input to a flat dict of OTel span attributes."""

    def map_llm_call(self, data: LLMCallSpanData) -> AttributeMap: ...
