from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.tracing.normalizers.base import to_int
from litellm.tracing.types import SpanRow, SpanType

_OPENINFERENCE_TYPES: Final[Mapping[str, SpanType]] = MappingProxyType({"AGENT": "agent", "LLM": "llm", "TOOL": "tool"})


@dataclass(frozen=True, slots=True)
class OpenInferenceNormalizer:
    name: str = "openinference"

    def matches(self, scope_name: str, attributes: Mapping[str, str]) -> bool:
        return "openinference.span.kind" in attributes

    def normalize(self, row: SpanRow, attributes: Mapping[str, str]) -> None:
        kind: Final = attributes.get("openinference.span.kind", "").upper()
        row["ObservationType"] = _OPENINFERENCE_TYPES.get(kind, "agent" if not row["ParentSpanId"] else "chain")
        row["AgentName"] = attributes.get("agent.name", "")
        row["Model"] = attributes.get("llm.model_name", "")
        row["Input"] = attributes.get("input.value", "")
        row["Output"] = attributes.get("output.value", "")
        row["InputTokens"] = to_int(attributes.get("llm.token_count.prompt"))
        row["OutputTokens"] = to_int(attributes.get("llm.token_count.completion"))
