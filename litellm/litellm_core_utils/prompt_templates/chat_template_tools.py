import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ToolChoiceTokens:
    none: int
    required: int
    named: int


OPENAI_TOOL_CHOICE_TOKENS: Final = ToolChoiceTokens(none=1, required=0, named=7)
_XAI_GROK_TOOL_CHOICE_TOKENS: Final = ToolChoiceTokens(none=21, required=11, named=12)
_XAI_SERVED_TEMPLATED_GROK: Final = re.compile(
    r"^(?:xai/|(?:openrouter/)?x-ai/)?grok-4[-.][13](?![.\d])", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class ChatTemplateToolSection:
    preamble_tokens: int
    implicit_system_turn_tokens: int
    render_tool_schemas: Callable[[Sequence[Mapping[str, object]]], str]
    renders_strict_only_when_true: bool = False


def _json_line_tool_schemas(functions: Sequence[Mapping[str, object]]) -> str:
    return "\n".join(json.dumps(function, ensure_ascii=False) for function in functions)


def _compact_json_line_tool_schemas(functions: Sequence[Mapping[str, object]]) -> str:
    return "\n".join(json.dumps(function, ensure_ascii=False, separators=(",", ":")) for function in functions)


def _qwen3_5_tool_schemas(functions: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        "\n" + json.dumps({"type": "function", "function": function}, ensure_ascii=False) for function in functions
    )


_TOOL_SECTIONS: Final = (
    (
        re.compile(r"deepseek-v4(?:[-._]|p)1(?!\d)", re.IGNORECASE),
        ChatTemplateToolSection(211, 0, _json_line_tool_schemas),
    ),
    (
        re.compile(
            r"deepseek-v4(?![.\d]|p\d)|^(?:deepseek/)?deepseek-(?:chat|reasoner|flash)(?::free)?$", re.IGNORECASE
        ),
        ChatTemplateToolSection(222, 0, _json_line_tool_schemas),
    ),
    (re.compile(r"qwen3[._]?5(?!\d)", re.IGNORECASE), ChatTemplateToolSection(196, 5, _qwen3_5_tool_schemas)),
    (
        re.compile(r"grok-4[-.]1(?![.\d])", re.IGNORECASE),
        ChatTemplateToolSection(
            0,
            0,
            _compact_json_line_tool_schemas,
            renders_strict_only_when_true=True,
        ),
    ),
    (
        re.compile(r"grok-4[-.]3(?![.\d])", re.IGNORECASE),
        ChatTemplateToolSection(
            0,
            0,
            _json_line_tool_schemas,
            renders_strict_only_when_true=True,
        ),
    ),
)


def chat_template_tool_section(model: str | None) -> ChatTemplateToolSection | None:
    if not model:
        return None
    return next((section for pattern, section in _TOOL_SECTIONS if pattern.search(model)), None)


def chat_template_tool_choice_tokens(model: str) -> ToolChoiceTokens | None:
    return _XAI_GROK_TOOL_CHOICE_TOKENS if _XAI_SERVED_TEMPLATED_GROK.match(model) else None
