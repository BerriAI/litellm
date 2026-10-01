"""LangSmith OTEL mode, which LangChain, LangGraph and Deep Agents export through."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.tracing.normalizers.messages import lc_message
from litellm.tracing.types import SpanRow, SpanType

# LangChain / Deep Agents middleware wrappers: real spans, but noise in the UI
_FRAMEWORK_SUFFIXES: Final = (
    ".wrap_model_call",
    ".wrap_tool_call",
    ".before_agent",
    ".after_agent",
    ".before_model",
    ".after_model",
)


def _loads(value: str) -> object:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _span_type(row: SpanRow, attributes: Mapping[str, str]) -> SpanType:
    kind: Final = attributes.get("langsmith.span.kind", "chain")
    name: Final = row["SpanName"]
    if not row["ParentSpanId"] or name == attributes.get("langsmith.metadata.lc_agent_name"):
        return "agent"
    if kind in ("llm", "tool"):
        return kind
    if name.endswith(_FRAMEWORK_SUFFIXES):
        return "framework"
    return "chain"


def _tool_output(completion: object) -> object:
    raw: Final = completion.get("output", completion) if isinstance(completion, dict) else completion
    update: Final = raw.get("update") if isinstance(raw, dict) else None
    update_messages: Final = update.get("messages") or () if isinstance(update, dict) else ()
    is_command: Final = isinstance(raw, dict) and "update" in raw
    # LangGraph Command (e.g. the Deep Agents `task` tool): the result is the last update message
    output: Final = update_messages[-1] if is_command and update_messages else raw
    return output.get("content", output) if isinstance(output, dict) else output


def _set_agent_io(row: SpanRow, attributes: Mapping[str, str], prompt: object, completion: object) -> None:
    input_messages: Final = prompt.get("messages") if isinstance(prompt, dict) else None
    output_messages: Final = completion.get("messages") if isinstance(completion, dict) else None
    # agents built with @traceable take arbitrary args, not a message list: keep the raw payload then
    row["Input"] = (
        json.dumps(tuple(lc_message(m) for m in input_messages if isinstance(m, dict)))
        if input_messages
        else attributes.get("gen_ai.prompt", "")
    )
    row["Output"] = (
        json.dumps(lc_message(output_messages[-1]))
        if output_messages and isinstance(output_messages[-1], dict)
        else attributes.get("gen_ai.completion", "")
    )


def _set_io(row: SpanRow, attributes: Mapping[str, str]) -> None:
    prompt: Final = _loads(attributes.get("gen_ai.prompt", ""))
    completion: Final = _loads(attributes.get("gen_ai.completion", ""))
    if row["ObservationType"] == "llm" and isinstance(completion, dict):
        prompt_payload: Final = prompt if isinstance(prompt, dict) else MappingProxyType({})
        messages: Final = prompt_payload.get("messages") or ((),)
        batch: Final = messages[0] if messages and isinstance(messages[0], list) else messages
        row["Input"] = (
            json.dumps(tuple(lc_message(m) for m in batch if isinstance(m, dict)))
            if isinstance(batch, (list, tuple))
            else ""
        )
        generations: Final = completion.get("generations")
        first: Final = generations[0] if isinstance(generations, list) and generations else None
        item: Final = first[0] if isinstance(first, list) and first else None
        message: Final = item.get("message") if isinstance(item, dict) else None
        generation: Final = message.get("kwargs") if isinstance(message, dict) else None
        if isinstance(generation, dict):
            row["Output"] = json.dumps(lc_message(generation))
            metadata: Final = generation.get("response_metadata")
            row["LiteLLMRequestId"] = metadata.get("id", "") if isinstance(metadata, dict) else ""
            return
        row["Output"] = attributes.get("gen_ai.completion", "")
        return
    if row["ObservationType"] == "tool":
        output: Final = _tool_output(completion)
        row["Input"] = attributes.get("gen_ai.prompt", "")
        row["Output"] = output if isinstance(output, str) else json.dumps(output)
        return
    if row["ObservationType"] == "agent":
        _set_agent_io(row, attributes, prompt, completion)
        return
    row["Input"] = attributes.get("gen_ai.prompt", "")
    row["Output"] = attributes.get("gen_ai.completion", "")


@dataclass(frozen=True, slots=True)
class LangSmithNormalizer:
    name: str = "langsmith"

    def matches(self, scope_name: str, attributes: Mapping[str, str]) -> bool:
        return scope_name == "langsmith" or "langsmith.span.kind" in attributes

    def normalize(self, row: SpanRow, attributes: Mapping[str, str]) -> None:
        row["ObservationType"] = _span_type(row, attributes)
        row["AgentName"] = attributes.get("langsmith.metadata.lc_agent_name", "")
        row["Model"] = attributes.get("gen_ai.request.model", "")
        _set_io(row, attributes)
