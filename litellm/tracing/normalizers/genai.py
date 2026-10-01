from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.tracing.types import SpanRow

_LLM_OPERATIONS: Final = frozenset({"chat", "text_completion", "generate_content"})


@dataclass(frozen=True, slots=True)
class GenAISemconvNormalizer:
    """OTEL `gen_ai.*` semantic conventions. Matches every span, so it belongs last as the fallback."""

    name: str = "genai"

    def matches(self, scope_name: str, attributes: Mapping[str, str]) -> bool:
        return True

    def normalize(self, row: SpanRow, attributes: Mapping[str, str]) -> None:
        operation: Final = attributes.get("gen_ai.operation.name", "")
        if operation == "invoke_agent" or not row["ParentSpanId"]:
            row["ObservationType"] = "agent"
        elif operation in _LLM_OPERATIONS:
            row["ObservationType"] = "llm"
        elif operation == "execute_tool":
            row["ObservationType"] = "tool"
        row["AgentName"] = attributes.get("gen_ai.agent.name", "")
        row["Model"] = attributes.get("gen_ai.request.model") or attributes.get("gen_ai.response.model", "")
        row["LiteLLMRequestId"] = attributes.get("gen_ai.response.id", "")
        row["Input"] = attributes.get("gen_ai.input.messages") or attributes.get("gen_ai.tool.call.arguments", "")
        row["Output"] = attributes.get("gen_ai.output.messages") or attributes.get("gen_ai.tool.call.result", "")
