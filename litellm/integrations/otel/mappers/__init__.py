"""Attribute mappers: pure ``LLMCallSpanData -> {attribute key: value}`` functions.

Composition over inheritance: vocabularies layer onto the same span. Listing
``["genai", "openinference"]`` in ``config.mapper_names`` makes every span
carry both the canonical ``gen_ai.*`` keys and the OpenInference (Arize +
Phoenix) keys. Add ``"langfuse"`` and it works for all three backends at once.
"""

from typing import Callable, Iterable

from litellm.integrations.otel.mappers.base import (
    AttributeMap,
    AttributeMapper,
    AttrValue,
)
from litellm.integrations.otel.mappers.genai import GenAIMapper
from litellm.integrations.otel.mappers.langfuse import LangfuseMapper
from litellm.integrations.otel.mappers.langtrace import LangtraceMapper
from litellm.integrations.otel.mappers.legacy import LegacyMapper
from litellm.integrations.otel.mappers.openinference import OpenInferenceMapper
from litellm.integrations.otel.mappers.utils import tool_attr_budget
from litellm.integrations.otel.mappers.weave import WeaveMapper

# Registries keyed by ``config.mapper_names`` entries, split by whether the
# vocabulary spells declared tool definitions out per index. Those share one
# span-wide attribute ceiling, so resolution has to know how many of them are
# active before it can build them.
_TOOL_DEFINITION_MAPPERS: dict[str, Callable[[int], AttributeMapper]] = {
    "genai": GenAIMapper,
    "legacy": LegacyMapper,
    "openinference": OpenInferenceMapper,
}
_PLAIN_MAPPERS: dict[str, Callable[[], AttributeMapper]] = {
    "langfuse": LangfuseMapper,
    "weave": WeaveMapper,
    "langtrace": LangtraceMapper,
}


def resolve_mappers(names: Iterable[str]) -> list[AttributeMapper]:
    """Resolve mapper names to instances. Unknown names raise ``ValueError``."""
    ordered = tuple(names)
    for name in ordered:
        if name not in _TOOL_DEFINITION_MAPPERS and name not in _PLAIN_MAPPERS:
            known = sorted((*_TOOL_DEFINITION_MAPPERS, *_PLAIN_MAPPERS))
            raise ValueError(f"unknown mapper name {name!r}; known: {known}")
    # Distinct vocabularies each write the tool family under their own keys, so
    # the ceiling is split by how many of them are configured. Repeating a name
    # rewrites the same keys, so only distinct ones count.
    budget = tool_attr_budget(len({*ordered} & _TOOL_DEFINITION_MAPPERS.keys()))
    return [
        _TOOL_DEFINITION_MAPPERS[name](budget) if name in _TOOL_DEFINITION_MAPPERS else _PLAIN_MAPPERS[name]()
        for name in ordered
    ]


__all__ = [
    "AttributeMap",
    "AttributeMapper",
    "AttrValue",
    "GenAIMapper",
    "LangfuseMapper",
    "LangtraceMapper",
    "LegacyMapper",
    "OpenInferenceMapper",
    "WeaveMapper",
    "resolve_mappers",
]
