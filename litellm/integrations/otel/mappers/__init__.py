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
from litellm.integrations.otel.mappers.weave import WeaveMapper
from litellm.integrations.otel.model.config import OpenTelemetryV2Config

# Registry keyed by ``config.mapper_names`` entries.
_MAPPER_BY_NAME: dict[str, Callable[[OpenTelemetryV2Config | None], AttributeMapper]] = {
    "genai": lambda _config: GenAIMapper(),
    "legacy": lambda _config: LegacyMapper(),
    "openinference": lambda _config: OpenInferenceMapper(),
    "langfuse": lambda config: LangfuseMapper(
        trace_metadata_keys=config.langfuse_trace_metadata_keys if config else ()
    ),
    "weave": lambda _config: WeaveMapper(),
    "langtrace": lambda _config: LangtraceMapper(),
}


def resolve_mappers(names: Iterable[str], config: OpenTelemetryV2Config | None = None) -> list[AttributeMapper]:
    """Resolve mapper names to instances. Unknown names raise ``ValueError``.

    ``config`` is optional so a caller that only wants a vocabulary's default
    behaviour (tests, ad-hoc mapping) needn't build one.
    """
    out: list[AttributeMapper] = []
    for name in names:
        factory = _MAPPER_BY_NAME.get(name)
        if factory is None:
            raise ValueError(f"unknown mapper name {name!r}; known: {sorted(_MAPPER_BY_NAME)}")
        out.append(factory(config))
    return out


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
