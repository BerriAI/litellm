import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ChatTemplateToolSection:
    preamble_tokens: int
    implicit_system_turn_tokens: int
    render_tool_schemas: Callable[[Sequence[Mapping[str, object]]], str]


def _deepseek_v4_tool_schemas(functions: Sequence[Mapping[str, object]]) -> str:
    return "\n".join(json.dumps(function, ensure_ascii=False) for function in functions)


def _qwen3_5_tool_schemas(functions: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        "\n" + json.dumps({"type": "function", "function": function}, ensure_ascii=False) for function in functions
    )


_TOOL_SECTIONS: Final = (
    (
        re.compile(
            r"deepseek-v4(?![.\d]|p\d)|^(?:deepseek/)?deepseek-(?:chat|reasoner|flash)(?::free)?$", re.IGNORECASE
        ),
        ChatTemplateToolSection(222, 0, _deepseek_v4_tool_schemas),
    ),
    (re.compile(r"qwen3[._]?5(?!\d)", re.IGNORECASE), ChatTemplateToolSection(196, 5, _qwen3_5_tool_schemas)),
)


def chat_template_tool_section(model: str | None) -> ChatTemplateToolSection | None:
    if not model:
        return None
    return next((section for pattern, section in _TOOL_SECTIONS if pattern.search(model)), None)
